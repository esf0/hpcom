"""How much of the noiseless split-step distortion do the neighbouring symbols explain?

For one power file of ``run.py`` (the last 8 runs test, the rest train), the
fraction of var(y_clean − x) explained, on the test runs, by

* the conditional mean given the transmitted triplet (qₜ₋₁, qₜ, qₜ₊₁) — all the
  deterministic information the paper's triplet model can use;
* a first-order perturbation model, Σ C_mn x_{t+m} x_{t+n} x*_{t+m+n} over
  |m|, |n| ≤ M, least squares on 6 training runs;

and the BER of symbol-wise decisions after a genie subtracts that prediction
from the noisy output (the true symbols are known): an upper bound on what a
decoder with that much memory can gain.

Run::

    python memory.py data/p+1.0.npz
"""

from __future__ import annotations

import sys

import numpy as np
from analyse import ber, smd, triplet_ids


def perturbation_features(x: np.ndarray, m_max: int) -> np.ndarray:
    cols = [x]
    for a in range(-m_max, m_max + 1):
        for b in range(-m_max, m_max + 1):
            cols.append(np.roll(x, -a) * np.roll(x, -b) * np.conj(np.roll(x, -(a + b))))
    return np.stack(cols, axis=-1)


def main() -> None:
    d = np.load(sys.argv[1])
    points, idx = d["points"], d["idx"].astype(np.int64)
    x = points[idx]
    yc, yn = d["y_clean"].astype(np.complex128), d["y_noisy"].astype(np.complex128)
    n_runs = len(idx)
    tr, te = slice(0, n_runs - 8), slice(n_runs - 8, None)
    dist = yc - x
    v_ss = np.mean(abs(dist[te]) ** 2)

    def line(name: str, pred: np.ndarray) -> None:
        explained = 1 - np.mean(abs(dist[te] - pred) ** 2) / v_ss
        print(
            f"{name:34s} explains {explained:6.1%}   genie SMD BER {ber(smd(yn[te] - pred, points), idx[te]):.3e}"
        )

    print(
        f"{float(d['power_dbm']):+.1f} dBm  SNR_ss {-10 * np.log10(v_ss):.2f} dB  "
        f"SNR_noise {-10 * np.log10(np.mean(abs(yn[te] - yc[te]) ** 2)):.2f} dB  "
        f"SMD BER {ber(smd(yn[te], points), idx[te]):.3e}  "
        f"(all distortion removed: {ber(smd(x[te] + yn[te] - yc[te], points), idx[te]):.3e})"
    )
    tri = triplet_ids(idx)
    counts = np.bincount(tri[tr].ravel(), minlength=4096)
    mean = (
        np.bincount(tri[tr].ravel(), weights=dist[tr].ravel().real, minlength=4096)
        + 1j * np.bincount(tri[tr].ravel(), weights=dist[tr].ravel().imag, minlength=4096)
    ) / counts
    line("triplet conditional mean", mean[tri[te]])
    n_fit = 6
    for m_max in (1, 2, 5, 10, 20):
        f = np.concatenate([perturbation_features(x[r], m_max) for r in range(n_fit)]).astype(
            np.complex64
        )
        c, *_ = np.linalg.lstsq(f, np.concatenate(dist[:n_fit]).astype(np.complex64), rcond=None)
        del f
        pred = np.stack([perturbation_features(x[r], m_max) @ c for r in range(n_runs - 8, n_runs)])
        line(f"perturbation |m|,|n| <= {m_max:2d} ({(2 * m_max + 1) ** 2 + 1} terms)", pred)


if __name__ == "__main__":
    main()
