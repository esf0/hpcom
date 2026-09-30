"""End-to-end learning of a neural-network encoder and decoder through a fiber link.

Every method is trained by gradient descent (Adam) *through the physical
channel* — RRC pulse shaping, 10 × 80 km of SSMF solved by the split-step
NLSE with EDFA noise, CD compensation, matched filter, data-aided phase
recovery — with ``hpcom.autodiff``, a fresh block of data and noise at every
step. Methods, per launch power:

* ``qam+gauss``  — 64-QAM, Gaussian (AWGN-metric) demapper: the baseline;
* ``qam+nn``     — 64-QAM, NN decoder with a ±K-symbol window;
* ``gs+gauss``   — 64 learned points (geometric shaping), Gaussian demapper;
* ``ae``         — NN encoder + memoryless NN decoder (an autoencoder);
* ``ae+mem``     — NN encoder + NN decoder with a ±K-symbol window;
* ``qam+dbp``    — QAM, digital back-propagation (2 samples/symbol, 4
  steps/span) and the Gaussian demapper: the classical nonlinearity
  compensation (not trained);
* ``gs+dbp``, ``ae+mem+dbp`` — learned points, or NN encoder + NN decoder
  (±K), trained *through* that DBP receiver.

NN designs (model-based + learned, so training starts from the classical
system and can only learn beyond it): the encoder is 64-QAM plus an MLP
correction of every point (one-hot → 64 → 2); the decoder's logits are a
Gaussian metric with a learned scale, :math:`-e^{s}|z_k - x_m|^2` (initially
:math:`1/\\sigma^2` measured on one block), plus an
MLP (window features → 128 → 128 → 64) of Re/Im of
:math:`z_{k-K}…z_{k+K}` and :math:`|z_k|^2`.

Rates are symbol-wise achievable-rate estimates on fresh data:
the Gaussian-channel MI of ``hpcom.metrics`` for the Gaussian demapper, and
the cross-entropy bound :math:`H(X) - \\mathrm{CE}/\\ln 2` for NN decoders —
both lower bounds on the mutual information for their decoding metric.

Run::

    python run.py --out results            # full sweep (GPU recommended)
    python run.py --quick --out smoke      # seconds, CPU is fine
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hpcom
from hpcom import (
    EDFA,
    ConstellationConfig,
    DigitalBackPropagation,
    Fiber,
    Link,
    SingleCarrierTransmitter,
)
from hpcom import autodiff as ad

M = 64  # constellation size; set by --order


# ---------------------------------------------------------------------------
# Small MLPs and Adam in plain JAX
# ---------------------------------------------------------------------------


def mlp_init(key: jax.Array, sizes: list[int], dtype) -> list[tuple[jax.Array, jax.Array]]:
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, itertools.pairwise(sizes), strict=True):
        w = jax.random.normal(k, (n_in, n_out), dtype) * math.sqrt(2.0 / (n_in + n_out))
        params.append((w, jnp.zeros(n_out, dtype)))
    return params


def mlp(params, x: jax.Array) -> jax.Array:
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


def adam_step(params, grads, state, lr: float):
    """Gradient *ascent* (the objective is a rate); complex leaves take the conjugate."""
    t = state["t"] + 1
    g = jax.tree_util.tree_map(jnp.conj, grads)
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, state["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * jnp.abs(b) ** 2, state["v"], g)
    new = jax.tree_util.tree_map(
        lambda p, a, b: p + lr * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8),
        params,
        m,
        v,
    )
    return new, {"m": m, "v": v, "t": t}


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


def system(args, steps_per_span: int | None = None):
    fiber = Fiber.ssmf(80.0, n_steps=steps_per_span or args.steps_per_span)
    link = Link.from_spans(args.spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = SingleCarrierTransmitter(
        constellation=ConstellationConfig.from_square_qam(M),
        symbol_rate_hz=32e9,
        n_symbols=args.symbols,
        samples_per_symbol=4,
    )
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
    return tx, link, rx


def window(z: jax.Array, half: int) -> jax.Array:
    """Decoder features: Re/Im of z[k−K … k+K] (circular) and |z[k]|², shape (n, 4K+3)."""
    cols = [jnp.roll(z, -j) for j in range(-half, half + 1)]
    feats = [c.real for c in cols] + [c.imag for c in cols] + [jnp.abs(z) ** 2]
    return jnp.stack(feats, axis=-1)


def make_method(name: str, args, tx, link, rx, key):
    """(params, rate_fn(params, key) → rate, learning rate) for a method."""
    rdtype = jnp.float32 if args.dtype == "c64" else jnp.float64
    cdtype = jnp.complex64 if args.dtype == "c64" else jnp.complex128
    qam = jnp.asarray(tx.resolved_constellation.points, cdtype)
    half = args.window if name in ("qam+nn", "ae+mem", "ae+mem+dbp") else 0
    k_enc, k_dec = jax.random.split(key)
    params: dict = {}
    if name in ("ae", "ae+mem", "ae+mem+dbp"):
        enc = mlp_init(k_enc, [M, 64, 2], rdtype)
        params["enc"] = [(w, b) for w, b in enc[:-1]] + [(enc[-1][0] * 0.01, enc[-1][1])]
    if name in ("gs+gauss", "gs+dbp"):
        params["points"] = qam
    if name in ("qam+nn", "ae", "ae+mem", "ae+mem+dbp"):
        dec = mlp_init(k_dec, [4 * half + 3, 128, 128, M], rdtype)
        params["dec"] = [(w, b) for w, b in dec[:-1]] + [(dec[-1][0] * 0.01, dec[-1][1])]
        params["log_scale"] = jnp.asarray(0.0, rdtype)  # set from the measured noise below
    power = jnp.asarray(args.power, rdtype)

    def points_of(p):
        if "enc" in p:
            out = mlp(p["enc"], jnp.eye(M, dtype=rdtype))
            pts = qam + (out[:, 0] + 1j * out[:, 1]).astype(cdtype)
        elif "points" in p:
            pts = p["points"]
        else:
            pts = qam
        return pts / jnp.sqrt(jnp.mean(jnp.abs(pts) ** 2))

    receiver = rx
    if name.endswith("dbp"):
        dbp = DigitalBackPropagation(link=link, steps_per_span=4, samples_per_symbol=2)
        receiver = rx.replace(cd_compensation_s2=0.0, compensation=dbp)

    def received(p, key, channel=link):
        k_data, k_noise = jax.random.split(key)
        pts = points_of(p)
        idx = jax.random.randint(k_data, (tx.n_symbols,), 0, M)
        x = ad.modulate(pts[idx], tx, launch_power_dbm=power)
        y = ad.propagate_link(x, channel, tx.dt_s, key=k_noise, gradient=args.gradient)
        return pts, idx, ad.demodulate(y, receiver, tx.dt_s, reference=pts[idx])

    if "log_scale" in params:  # start exactly at the Gaussian demapper: scale 1/σ² measured
        pts0, idx0, z0 = jax.jit(received)(params, jax.random.key(7))
        sigma2 = jnp.mean(jnp.abs(z0 - pts0[idx0]) ** 2)
        params["log_scale"] = -jnp.log(sigma2).astype(rdtype)

    def rate(p, key, channel=link):
        pts, idx, z = received(p, key, channel)
        if "dec" not in p:
            return ad.information_rates(z, idx, pts, tx.resolved_constellation.bit_labels)[0]
        metric = -jnp.exp(p["log_scale"]) * jnp.abs(z[:, None] - pts[None, :]) ** 2
        logits = metric + mlp(p["dec"], window(z, half))
        ce = jnp.mean(
            jax.nn.logsumexp(logits, axis=-1)
            - jnp.take_along_axis(logits, idx[:, None], axis=-1)[:, 0]
        )
        return math.log2(M) - ce / math.log(2.0)

    lr = 1e-2 if name.startswith("gs") else args.lr
    return params, rate, points_of, lr


def train_and_evaluate(name: str, args, tx, link, rx, seed: int) -> dict:
    """Train on fresh blocks; evaluate on ``eval_blocks`` blocks shared by every method (paired)."""
    k_init, k_train = jax.random.split(jax.random.key(seed))
    k_eval = jax.random.key(int(1e6 + 10 * args.power))
    params, rate, points_of, lr = make_method(name, args, tx, link, rx, k_init)
    history = []
    t0 = time.perf_counter()
    if params:
        value_and_grad = jax.jit(jax.value_and_grad(rate))
        state = adam_init(params)
        for i, k in enumerate(jax.random.split(k_train, args.iters)):
            value, grads = value_and_grad(params, k)
            params, state = adam_step(params, grads, state, lr)
            if i % 25 == 0 or i == args.iters - 1:
                history.append((i, float(value)))
    evaluate = jax.jit(rate)
    rates = [float(evaluate(params, k)) for k in jax.random.split(k_eval, args.eval_blocks)]
    fine = {}
    if args.eval_steps_per_span:
        # the trained system, evaluated on a finer (reference) simulation of the same link
        fine_link = system(args, steps_per_span=args.eval_steps_per_span)[1]
        evaluate_fine = jax.jit(lambda p, k: rate(p, k, fine_link))
        fine_rates = [
            float(evaluate_fine(params, k)) for k in jax.random.split(k_eval, args.eval_blocks)
        ]
        fine = {"rates_fine": fine_rates, "rate_fine_mean": float(np.mean(fine_rates))}
    return {
        **fine,
        "method": name,
        "power_dbm": args.power,
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(np.std(rates) / math.sqrt(len(rates))),
        "rates": rates,
        "history": history,
        "points": np.asarray(points_of(params))
        .view(np.float64 if args.dtype == "c128" else np.float32)
        .tolist(),
        "train_s": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--powers", type=float, nargs="+", default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    parser.add_argument(
        "--methods", nargs="+", default=["qam+gauss", "qam+nn", "gs+gauss", "ae", "ae+mem"]
    )
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--symbols", type=int, default=4096)
    parser.add_argument("--spans", type=int, default=10)
    parser.add_argument("--steps-per-span", type=int, default=20)
    parser.add_argument("--window", type=int, default=3, help="decoder memory ±K symbols")
    parser.add_argument(
        "--eval-steps-per-span", type=int, default=0, help="also evaluate on a finer channel"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--snr-guess-db", type=float, default=18.0, help="initial decoder metric scale"
    )
    parser.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    parser.add_argument(
        "--gradient", choices=["store", "checkpoint", "reversible"], default="store"
    )
    parser.add_argument("--order", type=int, default=64, help="constellation size M (square QAM)")
    parser.add_argument("--append", action="store_true", help="add to an existing results.json")
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = parser.parse_args()
    global M
    M = args.order
    if args.quick:
        args.powers, args.iters, args.eval_blocks, args.symbols, args.spans = [0.0], 20, 2, 512, 2
    if args.dtype == "c128":
        jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    tx, link, rx = system(args)
    results = []
    if args.append and (args.out / "results.json").exists():
        results = json.loads((args.out / "results.json").read_text())["results"]
    for power in args.powers:
        args.power = power
        for m_i, name in enumerate(args.methods):
            r = train_and_evaluate(name, args, tx, link, rx, seed=1000 + m_i)
            results.append(r)
            fine = f"  fine {r['rate_fine_mean']:.4f}" if "rate_fine_mean" in r else ""
            print(
                f"{power:+5.1f} dBm  {name:10s}  rate {r['rate_mean']:.4f} ± {r['rate_std']:.4f} "
                f"b/sym{fine}  ({r['train_s']:.0f} s)",
                flush=True,
            )
            (args.out / "results.json").write_text(
                json.dumps(
                    {
                        "config": {
                            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                        },
                        "devices": [str(d) for d in jax.devices()],
                        "hpcom": hpcom.__version__,
                        "jax": jax.__version__,
                        "results": results,
                    },
                    indent=1,
                )
            )


if __name__ == "__main__":
    main()
