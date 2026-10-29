"""Statistics of the paired runs and the triplet decoder on true and on synthetic noisy data.

For every power file written by ``run.py`` (the last ``--test-runs`` runs test, the rest train):

1. **Noise budget**, unit-power constellation: σ²_lin (linear ASE), σ²_paper (the
   paper's channel noise: noisy − noiseless, each stream scaled by its own
   data-aided gain as in the legacy ``nonlinear_shift``), σ²_tot (the same
   difference under the noiseless gain, which adds the mean phase rotation the ASE
   power causes), σ²_ss (signal–signal distortion of the noiseless run), and the
   true error variance of the noisy run.
2. **Shape of the channel noise** per ring and per constellation point: radial and
   tangential variance, excess kurtosis, correlation with the deterministic distortion.
3. **Decisions.** Symbol-wise minimum distance (SMD) on
   A — the true noisy split-step output;
   B — noiseless output + AWGN of variance σ²_paper (the paper's data model);
   C — noiseless output + AWGN of variance σ²_lin (fixed OSNR, linear ASE only);
   D — noiseless output + AWGN of variance σ²_lin + σ²_ss (nonlinear distortion
   counted twice, as a GN-type power-dependent term would add it).
   Triplet decoder: exact forward–backward (marginal MAP) on the paper's factor
   graph ``∏ₜ p(zₜ | qₜ₋₁, qₜ, qₜ₊₁)`` — on an open chain this is the fixed point
   GBP computes — and the most probable sequence on the same trellis
   (Viterbi, ``*_viterbi``). Likelihoods: per-triplet GMM (EM) fitted on the noiseless
   training data with σ²I added (``paper``: σ²_paper; ``osnr``: σ²_lin; ``paper_k1``:
   one Gaussian), or fitted directly on the noisy training data (``oracle``).

Run::

    python analyse.py data --out results     # then: python plot.py results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hpcom import SingleCarrierTransmitter

LABELS = SingleCarrierTransmitter(
    modulation_order=16, symbol_rate_hz=1e9, n_symbols=16, samples_per_symbol=2
).resolved_constellation.bit_labels


def triplet_ids(idx: np.ndarray) -> np.ndarray:
    """Triplet id ``16²·qₜ₋₁ + 16·qₜ + qₜ₊₁`` for every symbol of circular runs ``(R, T)``."""
    i = idx.astype(np.int64)
    return 256 * np.roll(i, 1, axis=-1) + 16 * i + np.roll(i, -1, axis=-1)


def ber(decisions: np.ndarray, idx: np.ndarray) -> float:
    return float(np.mean(LABELS[decisions] != LABELS[idx]))


def awgn(rng: np.random.Generator, shape: tuple[int, ...], var: float) -> np.ndarray:
    return np.sqrt(var / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def smd(z: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(z[..., None] - points) ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Per-triplet Gaussian mixtures (batched EM over all 4096 triplets)
# ---------------------------------------------------------------------------


def fit_gmm(z: np.ndarray, tri: np.ndarray, k: int, iters: int = 150, reg: float = 1e-8) -> dict:
    """EM fit of a ``k``-component 2-D GMM to the points of every triplet.

    Returns ``mu (4096, k, 2)``, ``cov (4096, k, 2, 2)``, ``w (4096, k)``.
    """
    order = np.argsort(tri, kind="stable")
    t_sorted, z_sorted = tri[order], z[order]
    counts = np.bincount(t_sorted, minlength=4096)
    n_max = counts.max()
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(t_sorted)) - start[t_sorted]
    x = np.zeros((4096, n_max, 2))
    mask = np.zeros((4096, n_max))
    x[t_sorted, pos, 0], x[t_sorted, pos, 1] = z_sorted.real, z_sorted.imag
    mask[t_sorted, pos] = 1.0
    n = counts[:, None]
    mean = (x * mask[..., None]).sum(1) / n
    d = (x - mean[:, None]) * mask[..., None]
    cov = np.einsum("jni,jnk->jik", d, d) / n[..., None] + reg * np.eye(2)
    if k == 1:
        return {"mu": mean[:, None], "cov": cov[:, None], "w": np.ones((4096, 1))}
    # Initialise along the principal axis of the single Gaussian.
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, :, -1] * np.sqrt(evals[:, -1:])
    offsets = np.linspace(-1, 1, k)
    mu = mean[:, None] + offsets[None, :, None] * axis[:, None]
    covs = np.repeat(cov[:, None] / k, k, axis=1)
    w = np.full((4096, k), 1.0 / k)
    for _ in range(iters):
        inv = np.linalg.inv(covs)
        logdet = np.log(np.linalg.det(covs))
        diff = x[:, None] - mu[:, :, None]  # (J, k, n, 2)
        q = np.einsum("jkni,jkil,jknl->jkn", diff, inv, diff)
        logp = np.log(w)[..., None] - math.log(2 * math.pi) - 0.5 * logdet[..., None] - 0.5 * q
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r = r / r.sum(axis=1, keepdims=True) * mask[:, None]
        nk = r.sum(axis=2) + 1e-12
        w = nk / n
        mu = np.einsum("jkn,jni->jki", r, x) / nk[..., None]
        diff = x[:, None] - mu[:, :, None]
        covs = np.einsum("jkn,jkni,jknl->jkil", r, diff, diff) / nk[..., None, None] + reg * np.eye(
            2
        )
        w = np.maximum(w, 1e-6)
        w /= w.sum(axis=1, keepdims=True)
    return {"mu": mu, "cov": covs, "w": w}


# ---------------------------------------------------------------------------
# Exact forward–backward on the triplet chain (JAX)
# ---------------------------------------------------------------------------


@jax.jit
def _posteriors(
    z: jnp.ndarray, mu: jnp.ndarray, prec: jnp.ndarray, logc: jnp.ndarray
) -> jnp.ndarray:
    """Marginal posteriors ``(T−2, 16)`` of q₁ … q_{T−2} for one run ``z (T, 2)``."""

    def factor(zt: jnp.ndarray) -> jnp.ndarray:
        d = zt[None, None] - mu
        ll = logc - 0.5 * jnp.einsum("jki,jkil,jkl->jk", d, prec, d)
        ll = jax.scipy.special.logsumexp(ll, axis=1)
        return jnp.exp(jnp.maximum(ll - ll.max(), -700.0)).reshape(16, 16, 16)  # (l, c, r)

    def forward(f: jnp.ndarray, zt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        fn = jnp.einsum("lc,lcr->cr", f, factor(zt))
        return fn / fn.sum(), f

    def backward(
        b: jnp.ndarray, inp: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        zt, f = inp
        m = factor(zt)
        post = jnp.einsum("lc,lcr,cr->c", f, m, b)
        bn = jnp.einsum("lcr,cr->lc", m, b)
        return bn / bn.sum(), post / post.sum()

    flat = jnp.full((16, 16), 1.0 / 256)
    _, fs = jax.lax.scan(forward, flat, z[1:-1])
    _, posts = jax.lax.scan(backward, flat, (z[1:-1], fs), reverse=True)
    return posts


@jax.jit
def _viterbi(z: jnp.ndarray, mu: jnp.ndarray, prec: jnp.ndarray, logc: jnp.ndarray) -> jnp.ndarray:
    """Most probable sequence q₁ … q_{T−2} ``(T−2,)`` of one run ``z (T, 2)``: max-product on the same pair states."""

    def loglik(zt: jnp.ndarray) -> jnp.ndarray:
        d = zt[None, None] - mu
        ll = logc - 0.5 * jnp.einsum("jki,jkil,jkl->jk", d, prec, d)
        return jax.scipy.special.logsumexp(ll, axis=1).reshape(16, 16, 16)  # (l, c, r)

    def forward(delta: jnp.ndarray, zt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        s = delta[:, :, None] + loglik(zt)  # (l, c, r)
        return s.max(axis=0) - s.max(), jnp.argmax(s, axis=0).astype(jnp.uint8)

    def backtrack(state: jnp.ndarray, psi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        c, r = state[0], state[1]
        return jnp.stack([psi[c, r].astype(jnp.int32), c]), c

    delta, psis = jax.lax.scan(forward, jnp.zeros((16, 16)), z[1:-1])
    last = jnp.argmax(delta)
    _, path = jax.lax.scan(
        backtrack, jnp.stack([last // 16, last % 16]).astype(jnp.int32), psis, reverse=True
    )
    return path


def triplet_decode(
    z: np.ndarray, gmm: dict, extra_var: float, sequence: bool = False
) -> np.ndarray:
    """Decisions ``(R, T−2)`` for runs ``z (R, T)``; ``extra_var`` = complex AWGN variance added to every component.

    Symbol-wise MAP from the exact marginals, or with ``sequence`` the most probable sequence (Viterbi).
    """
    with jax.enable_x64(True):
        cov = gmm["cov"] + (extra_var / 2) * np.eye(2)
        prec = jnp.asarray(np.linalg.inv(cov))
        logc = jnp.asarray(
            np.log(gmm["w"]) - math.log(2 * math.pi) - 0.5 * np.log(np.linalg.det(cov))
        )
        mu = jnp.asarray(gmm["mu"])
        zz = jnp.asarray(np.stack([z.real, z.imag], axis=-1).astype(np.float64))
        if sequence:
            return np.asarray(jax.vmap(_viterbi, in_axes=(0, None, None, None))(zz, mu, prec, logc))
        posts = jax.vmap(_posteriors, in_axes=(0, None, None, None))(zz, mu, prec, logc)
        return np.asarray(jnp.argmax(posts, axis=-1))


# ---------------------------------------------------------------------------


def lag_corr(n: np.ndarray, lag: int) -> dict:
    """Correlation of ``n`` (runs × symbols, circular) with itself ``lag`` symbols later: complex and of |n|²."""
    m = np.roll(n, -lag, axis=-1)
    p, pm = np.abs(n) ** 2, np.abs(m) ** 2
    return {
        "complex": float(abs(np.vdot(n, m)) / np.vdot(n, n).real),
        "power": float(np.corrcoef(p.ravel(), pm.ravel())[0, 1]),
    }


def noise_shape(n: np.ndarray, x: np.ndarray) -> dict:
    """Radial / tangential variance (per quadrature) and excess kurtosis of ``n`` per ring of ``x``."""
    u = x / np.abs(x)
    rad, tan = (n * np.conj(u)).real, (n * np.conj(u)).imag
    out = {}
    for name, ring in zip(("inner", "middle", "outer"), (0.2, 1.0, 1.8), strict=True):
        sel = np.isclose(np.abs(x) ** 2, ring, atol=1e-3)
        r, t = rad[sel], tan[sel]
        out[name] = {
            "var_radial": float(r.var()),
            "var_tangential": float(t.var()),
            "kurt_radial": float(np.mean((r - r.mean()) ** 4) / r.var() ** 2 - 3),
            "kurt_tangential": float(np.mean((t - t.mean()) ** 4) / t.var() ** 2 - 3),
        }
    return out


def per_point(n: np.ndarray, idx: np.ndarray) -> dict:
    """Mean and variance of ``n`` conditioned on the transmitted point (16 values each)."""
    out: dict = {"mean": [], "var": []}
    for q in range(16):
        nq = n[idx == q]
        out["mean"].append([float(nq.mean().real), float(nq.mean().imag)])
        out["var"].append(float(np.mean(np.abs(nq - nq.mean()) ** 2)))
    return out


def analyse(path: Path, rng: np.random.Generator, k: int, n_test: int) -> dict:
    d = np.load(path)
    points, idx = d["points"], d["idx"]
    x = points[idx]
    yc, yn = d["y_clean"].astype(np.complex128), d["y_noisy"].astype(np.complex128)
    nt, nl = d["n_tot"].astype(np.complex128), d["n_lin"].astype(np.complex128)
    npap = yn - yc  # legacy: both streams already scaled by their own gain
    tr, te = slice(0, len(idx) - n_test), slice(len(idx) - n_test, None)
    dist = yc - x
    var = {
        "lin": float(np.mean(np.abs(nl[tr]) ** 2)),
        "paper": float(np.mean(np.abs(npap[tr]) ** 2)),
        "tot": float(np.mean(np.abs(nt[tr]) ** 2)),
        "ss": float(np.mean(np.abs(dist[tr]) ** 2)),
        "noisy": float(np.mean(np.abs(yn[tr] - x[tr]) ** 2)),
    }
    corr = float(
        np.real(np.vdot(dist[tr], npap[tr])) / math.sqrt(var["ss"] * var["paper"]) / dist[tr].size
    )
    row: dict = {
        "power_dbm": float(d["power_dbm"]),
        "var": var,
        "snr_db": {key: -10 * math.log10(v) for key, v in var.items()},
        "corr_distortion_noise": corr,
        "shape_n_paper": noise_shape(npap[tr].ravel(), x[tr].ravel()),
        "shape_n_lin": noise_shape(nl[tr].ravel(), x[tr].ravel()),
        "point_n_paper": per_point(npap[tr].ravel(), idx[tr].ravel()),
        "point_n_lin": per_point(nl[tr].ravel(), idx[tr].ravel()),
        # Symbol-to-symbol correlation of the noise: zero for white ASE after a matched
        # filter; the factorisation P(Z|Q) = ∏ P(zₜ|Q) needs it to vanish.
        "noise_corr_lag": {
            name: {lag: lag_corr(n[tr], lag) for lag in (1, 2, 3)}
            for name, n in (("n_paper", npap), ("n_lin", nl))
        },
    }
    yt, it = yc[te], idx[te]
    data = {
        "A_true": yn[te],
        "B_paper": yt + awgn(rng, yt.shape, var["paper"]),
        "C_osnr": yt + awgn(rng, yt.shape, var["lin"]),
        "D_double": yt + awgn(rng, yt.shape, var["lin"] + var["ss"]),
    }
    inner = (slice(None), slice(1, -1))  # the triplet decoder scores symbols 1 … T−2
    row["ber_smd"] = {key: ber(smd(z, points)[inner], it[inner]) for key, z in data.items()}

    tri = triplet_ids(idx[tr]).ravel()
    clean = {kk: fit_gmm(yc[tr].ravel(), tri, kk) for kk in sorted({1, k})}
    oracle = fit_gmm(yn[tr].ravel(), tri, k)
    decoders = {
        ("A_true", "paper"): (clean[k], var["paper"], False),
        ("A_true", "paper_viterbi"): (clean[k], var["paper"], True),
        ("A_true", "paper_k1"): (clean[1], var["paper"], False),
        ("A_true", "osnr"): (clean[k], var["lin"], False),
        ("A_true", "oracle"): (oracle, 0.0, False),
        ("B_paper", "paper"): (clean[k], var["paper"], False),
        ("B_paper", "paper_viterbi"): (clean[k], var["paper"], True),
        ("C_osnr", "osnr"): (clean[k], var["lin"], False),
    }
    row["ber_triplet"] = {}
    for (dkey, mkey), (gmm, extra, sequence) in decoders.items():
        dec = triplet_decode(data[dkey], gmm, extra, sequence)
        row["ber_triplet"][f"{dkey}/{mkey}"] = ber(dec, it[inner])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--test-runs", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data.glob("p*.npz"), key=lambda f: float(f.stem[1:]))
    rows = []
    for f in files:
        row = analyse(f, np.random.default_rng(7), args.components, args.test_runs)
        rows.append(row)
        s, b, t = row["snr_db"], row["ber_smd"], row["ber_triplet"]
        print(
            f"{row['power_dbm']:+5.1f} dBm  SNR lin {s['lin']:5.2f} paper {s['paper']:5.2f} tot {s['tot']:5.2f} "
            f"ss {s['ss']:5.2f} "
            f"noisy {s['noisy']:5.2f} dB | SMD A {b['A_true']:.3e} B {b['B_paper']:.3e} "
            f"C {b['C_osnr']:.3e} D {b['D_double']:.3e} | triplet A/paper {t['A_true/paper']:.3e} "
            f"A/viterbi {t['A_true/paper_viterbi']:.3e} B/paper {t['B_paper/paper']:.3e} A/oracle {t['A_true/oracle']:.3e} A/osnr {t['A_true/osnr']:.3e}",
            flush=True,
        )
        (args.out / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
