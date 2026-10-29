"""Legacy dataset of the paper against hpcom v1: same link (15 × 80 km, +1 dBm), same statistics?

The legacy test file (one run of 4096 symbols, written by
~/triplet-fitting/legacy/data_collection_pilot_and_channel_noise.py) holds
the transmitted, the noiseless and the noisy received symbols. For it and for
the run.py output: SNR of the noiseless distortion and of the channel noise
(own-gain difference), and the fraction of the distortion that linear ISI over
±5 symbols and first-order perturbation terms over ±M symbols explain
(two-fold cross-validated least squares).

Run (needs pandas for the legacy pickle)::

    python legacy_check.py ~/data/errorstat_channel_noise/data_collected_test_nch_1_pavedbm_1.0.pkl data/p+1.0.npz
"""

import sys

import numpy as np
import pandas as pd


def explained(x, dist, cols_fn, n_folds=2):
    """Cross-validated fraction of var(dist) explained by a linear fit on the features cols_fn(x)."""
    feats = cols_fn(x)
    n = len(x)
    fold = np.arange(n) * n_folds // n
    res = np.empty_like(dist)
    coefs = []
    for f in range(n_folds):
        trn, tst = fold != f, fold == f
        c, *_ = np.linalg.lstsq(feats[trn], dist[trn], rcond=None)
        res[tst] = dist[tst] - feats[tst] @ c
        coefs.append(c)
    return 1 - np.mean(abs(res) ** 2) / np.mean(abs(dist) ** 2), np.mean(coefs, axis=0)


def linear(k_max):
    return lambda x: np.stack([np.roll(x, -k) for k in range(-k_max, k_max + 1)], axis=-1)


def pert(m_max):
    def f(x):
        cols = []
        for a in range(-m_max, m_max + 1):
            for b in range(-m_max, m_max + 1):
                cols.append(np.roll(x, -a) * np.roll(x, -b) * np.conj(np.roll(x, -(a + b))))
        return np.stack(cols, axis=-1)

    return f


def report(name, x, yc, yn):
    s = np.sqrt(np.mean(abs(x) ** 2))
    x, yc, yn = x / s, yc / s, yn / s
    dist = yc - x
    print(
        f"== {name}: {len(x)} symbols  SNR_ss {-10 * np.log10(np.mean(abs(dist) ** 2)):.2f} dB  "
        f"SNR_noise {-10 * np.log10(np.mean(abs(yn - yc) ** 2)):.2f} dB  mean dist {abs(dist.mean()):.1e}"
    )
    fr, c = explained(x, dist, linear(5))
    print(
        f"   linear ISI k=-5..5: explains {fr:6.1%};  |h_k| = "
        + " ".join(f"{abs(v):.3f}" for v in c)
    )
    for m_max in (1, 2, 4):
        fr, _ = explained(x, dist, pert(m_max))
        print(f"   perturbation M={m_max}: explains {fr:6.1%}")


df = pd.read_pickle(sys.argv[1])
r = df.iloc[0]
sc = r.scale_coef
report(
    "legacy (paper data, 1 block)",
    np.asarray(r.points_orig[0]) * sc,
    np.asarray(r.points_shifted_wo_noise[0]) * sc,
    np.asarray(r.points_shifted[0]) * sc,
)

d = np.load(sys.argv[2])
x = d["points"][d["idx"]]
yc, yn = d["y_clean"].astype(np.complex128), d["y_noisy"].astype(np.complex128)
report("hpcom v1, 4096-symbol block", x[0, :4096], yc[0, :4096], yn[0, :4096])
report("hpcom v1, 65536-symbol run", x[0], yc[0], yn[0])
