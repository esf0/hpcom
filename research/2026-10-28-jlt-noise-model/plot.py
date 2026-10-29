"""Figure of ``analyse.py`` results: noise budget, per-symbol noise spread, decisions.

Run::

    python plot.py results --paper paper/fig3_bottom_15x80.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--paper", type=Path, help="CSV of the paper's BERs (columns power, SMD, GBP, Viterbi)"
    )
    parser.add_argument(
        "--zoom", type=float, nargs=2, metavar=("LO", "HI"), help="inset of the BER panel, dBm"
    )
    args = parser.parse_args()
    rows = json.loads((args.results / "results.json").read_text())
    p = [r["power_dbm"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for key, label in (
        ("lin", "linear ASE only, $1/\\sigma^2_{lin}$"),
        ("paper", "paper's channel noise (noisy − noiseless), $1/\\sigma^2_{paper}$"),
        ("ss", "noiseless SSFM distortion, $1/\\sigma^2_{ss}$"),
        ("noisy", "noisy SSFM, total"),
    ):
        ax[0].plot(p, [r["snr_db"][key] for r in rows], "o-", label=label)
    ax[0].set(
        xlabel="launch power [dBm]",
        ylabel="SNR [dB]",
        title="Noise budget (unit-power constellation)",
    )
    ax[0].legend(fontsize=7)

    for name, style in (("n_paper", "o-"), ("n_lin", "s--")):
        v = np.array([r[f"point_{name}"]["var"] for r in rows])
        ax[1].plot(p, v.max(1) / v.min(1), style, label=f"{name}: max/min over the 16 points")
    for ring, style in (("inner", "^:"), ("outer", "v:")):
        s = [r["shape_n_paper"][ring] for r in rows]
        ax[1].plot(
            p,
            [x["var_tangential"] / x["var_radial"] for x in s],
            style,
            label=f"n_paper {ring} ring: tangential/radial",
        )
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set(
        xlabel="launch power [dBm]",
        ylabel="variance ratio",
        title="Symbol dependence of the channel noise",
    )
    ax[1].legend(fontsize=7)

    for key, label, style in (
        ("A_true", "SMD, true noisy SSFM", "o-"),
        ("B_paper", "SMD, noiseless + AWGN($\\sigma^2_{paper}$)", "s--"),
        ("D_double", "SMD, noiseless + AWGN($\\sigma^2_{lin}+\\sigma^2_{ss}$)", "x:"),
    ):
        ax[2].semilogy(p, [r["ber_smd"][key] for r in rows], style, label=label)
    for key, label, style in (
        ("A_true/paper", "triplet marginals (paper model), true noisy SSFM", "o-"),
        ("A_true/paper_viterbi", "triplet Viterbi (paper model), true noisy SSFM", "v-"),
        ("B_paper/paper", "triplet marginals (paper model), noiseless + AWGN", "s--"),
        ("A_true/oracle", "triplet marginals (GMM fitted on noisy data), true", "^-"),
    ):
        if key in rows[0]["ber_triplet"]:
            ax[2].semilogy(p, [r["ber_triplet"][key] for r in rows], style, label=label)
    if args.paper:
        with args.paper.open() as f:
            ref = list(csv.DictReader(f))
        pp = [float(r.get("power_dbm") or r["Power"]) for r in ref]
        for key, style in (("SMD", "k+"), ("GBP", "kd"), ("Viterbi", "k*")):
            ax[2].semilogy(
                pp, [float(r[key]) for r in ref], style, mfc="none", label=f"paper Fig. 3: {key}"
            )
    if args.zoom:
        lo, hi = args.zoom
        inset = ax[2].inset_axes([0.55, 0.06, 0.42, 0.36])
        for line in ax[2].get_lines():
            inset.plot(
                *line.get_data(),
                line.get_linestyle(),
                marker=line.get_marker(),
                color=line.get_color(),
                mfc="none",
            )
        inset.set_yscale("log")
        inset.set_xlim(lo, hi)
        ys = [
            y
            for line in ax[2].get_lines()
            for x, y in zip(*line.get_data(), strict=True)
            if lo <= x <= hi
        ]
        inset.set_ylim(0.9 * min(ys), 1.1 * max(ys))
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.3)
    ax[2].set(xlabel="launch power [dBm]", ylabel="BER", title="Decisions")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.results / "noise_model.png", dpi=150)


if __name__ == "__main__":
    main()
