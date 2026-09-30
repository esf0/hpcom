"""Figures and a summary table from ``results/results.json`` (see ``run.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "qam+gauss": "QAM, Gaussian demapper",
    "qam+nn": "QAM, NN decoder (±3 symbols)",
    "gs+gauss": "learned points, Gaussian demapper",
    "ae": "NN encoder + NN decoder",
    "ae+mem": "NN encoder + NN decoder (±3 symbols)",
    "qam+dbp": "QAM, DBP (2 sps, 4 steps/span), Gaussian",
    "gs+dbp": "learned points, DBP, Gaussian",
    "ae+mem+dbp": "NN encoder, DBP, NN decoder (±3 symbols)",
}


def main(folder: Path) -> None:
    data = json.loads((folder / "results.json").read_text())
    rows = data["results"]
    methods = [m for m in LABELS if any(r["method"] == m for r in rows)]
    powers = sorted({r["power_dbm"] for r in rows})

    def get(method: str, power: float) -> dict:
        return next(r for r in rows if r["method"] == method and r["power_dbm"] == power)

    # rate vs power
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in methods:
        mean = [get(m, p)["rate_mean"] for p in powers]
        err = [get(m, p)["rate_std"] for p in powers]
        ax.errorbar(powers, mean, yerr=err, marker="o", capsize=3, label=LABELS[m])
    ax.set_xlabel("launch power per channel [dBm]")
    ax.set_ylabel("achievable rate [bit/symbol]")
    order = data["config"].get("order", 64)
    ax.set_title(f"M = {order}, 32 GBd, 10 × 80 km SSMF, EDFA NF 5 dB")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "rate_vs_power.png", dpi=150)

    # paired gains over the baseline
    base = {p: np.asarray(get("qam+gauss", p)["rates"]) for p in powers}
    lines = [
        "| method | " + " | ".join(f"{p:+.0f} dBm" for p in powers) + " | best |",
        "|---" * (len(powers) + 2) + "|",
    ]
    for m in methods:
        cells = []
        for p in powers:
            r = get(m, p)
            d = np.asarray(r["rates"]) - base[p]
            cells.append(f"{r['rate_mean']:.3f} ({d.mean():+.3f}±{d.std() / np.sqrt(d.size):.3f})")
        best = max(get(m, p)["rate_mean"] for p in powers)
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + f" | {best:.3f} |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # constellations at each method's best power
    shaped = [m for m in methods if m in ("gs+gauss", "ae+mem", "gs+dbp", "ae+mem+dbp")]
    fig, axes = plt.subplots(1, len(shaped) + 1, figsize=(3.2 * (len(shaped) + 1), 3.2))
    qam = np.asarray(get("qam+gauss", powers[0])["points"]).reshape(-1, 2)
    axes[0].scatter(qam[:, 0], qam[:, 1], s=8)
    axes[0].set_title(f"{data['config'].get('order', 64)}-QAM", fontsize=9)
    for ax, m in zip(axes[1:], shaped, strict=True):
        p_best = max(powers, key=lambda p, m=m: get(m, p)["rate_mean"])
        pts = np.asarray(get(m, p_best)["points"]).reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color="C3")
        ax.set_title(f"{LABELS[m]}\n{p_best:+.0f} dBm", fontsize=8)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "constellations.png", dpi=150)

    # training curves at the highest power
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    p_hi = powers[-1]
    for m in methods:
        h = np.asarray(get(m, p_hi)["history"])
        if h.size:
            ax.plot(h[:, 0], h[:, 1], label=LABELS[m], alpha=0.8)
    ax.axhline(
        get("qam+gauss", p_hi)["rate_mean"], color="k", ls="--", lw=1, label="QAM, Gaussian (eval)"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("batch rate [bit/symbol]")
    ax.set_title(f"training at {p_hi:+.0f} dBm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "training.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results"))
