"""Per-power summary of ``results.json`` as two Markdown tables: the channel noise, the decisions.

Run::

    python summary.py results > results/summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    rows = json.loads((Path(sys.argv[1]) / "results.json").read_text())
    print(
        "Channel noise (SNR in dB, unit-power constellation; noise = noisy − noiseless, own gains):\n"
    )
    print(
        "| P [dBm] | SNR lin ASE | SNR noise | SNR noiseless distortion | SNR noisy | σ²_noise/σ²_lin "
        "| per-point var max/min (noise / lin ASE) | outer ring tan/rad | max abs excess kurtosis | lag-1 corr |"
    )
    print("|" + "---|" * 10)
    for r in rows:
        v, vl = np.array(r["point_n_paper"]["var"]), np.array(r["point_n_lin"]["var"])
        sh = r["shape_n_paper"]
        kurt = max(abs(sh[k][q]) for k in sh for q in ("kurt_radial", "kurt_tangential"))
        lag = r["noise_corr_lag"]["n_paper"]["1"]
        s = r["snr_db"]
        print(
            f"| {r['power_dbm']:+.1f} | {s['lin']:.2f} | {s['paper']:.2f} | {s['ss']:.2f} | {s['noisy']:.2f} "
            f"| {r['var']['paper'] / r['var']['lin']:.3f} | {v.max() / v.min():.3f} / {vl.max() / vl.min():.3f} "
            f"| {sh['outer']['var_tangential'] / sh['outer']['var_radial']:.3f} | {kurt:.3f} "
            f"| {max(lag['complex'], abs(lag['power'])):.4f} |"
        )
    print("\nBER (gain = BER reduction against SMD on the true noisy output):\n")
    print(
        "| P [dBm] | SMD, true noisy | SMD, noiseless + AWGN(σ²_noise) | SMD, noiseless + AWGN(σ²_lin + σ²_ss) "
        "| triplet marginals, true (gain) | triplet Viterbi, true (gain) | triplet marginals, noiseless + AWGN "
        "| triplet, GMM fitted on noisy | triplet, 1 Gaussian |"
    )
    print("|" + "---|" * 9)
    for r in rows:
        b, t = r["ber_smd"], r["ber_triplet"]

        def gain(key: str, t=t, b=b) -> str:
            return f"{t[key]:.3e} ({1 - t[key] / b['A_true']:.1%})" if key in t else "–"

        print(
            f"| {r['power_dbm']:+.1f} | {b['A_true']:.3e} | {b['B_paper']:.3e} | {b['D_double']:.3e} "
            f"| {gain('A_true/paper')} | {gain('A_true/paper_viterbi')} | {t['B_paper/paper']:.3e} "
            f"| {t['A_true/oracle']:.3e} | {t['A_true/paper_k1']:.3e} |"
        )


if __name__ == "__main__":
    main()
