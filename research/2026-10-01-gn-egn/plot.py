"""η vs spans per format, and model errors relative to the simulation (see ``run.py``)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TITLES = {"qpsk": "PM-QPSK", "16qam": "PM-16QAM", "64qam": "PM-64QAM", "gaussian": "PM-Gaussian"}
MODELS = {
    "gn": ("GN", {"color": "C0", "ls": "-."}),
    "gn_incoherent": ("GN, incoherent", {"color": "C0", "ls": ":"}),
    "egn": ("EGN, closed-form correction", {"color": "C2", "ls": "-"}),
    "first_order": (
        "first-order perturbation",
        {"color": "C3", "ls": "--", "marker": "s", "ms": 4},
    ),
}


def db(x: float | None) -> float:
    return math.nan if x is None else 10 * math.log10(x)


def main(folder: Path) -> None:
    rows = json.loads((folder / "results.json").read_text())["rows"]
    formats = [f for f in TITLES if any(r["format"] == f for r in rows)]
    fig, axes = plt.subplots(2, len(formats), figsize=(3.6 * len(formats), 6.6), sharex=True)
    lines = [
        "| format | spans | GN | GN incoh. | EGN closed form | first order | SSFM | SSFM +3 dB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for j, fmt in enumerate(formats):
        sel = sorted((r for r in rows if r["format"] == fmt), key=lambda r: r["spans"])
        n = np.array([r["spans"] for r in sel])
        sim = np.array([db(r["ssfm"]) for r in sel])
        top, bottom = axes[0, j], axes[1, j]
        top.errorbar(
            n,
            sim,
            yerr=[4.343 * r["ssfm_se"] / r["ssfm"] for r in sel],
            fmt="ko",
            ms=5,
            label="split-step simulation",
            zorder=5,
        )
        for key, (label, style) in MODELS.items():
            vals = np.array([db(r[key]) for r in sel])
            if np.all(np.isnan(vals)):
                continue
            top.plot(n, vals, label=label, **style)
            bottom.plot(n, vals - sim, label=label, **style)
        bottom.axhline(0.0, color="k", lw=0.8)
        top.set_title(TITLES[fmt])
        top.set_xscale("log")
        bottom.set_xlabel("spans of 80 km")
        top.grid(alpha=0.3, which="both")
        bottom.grid(alpha=0.3, which="both")
        for r in sel:
            lines.append(
                f"| {TITLES[fmt]} | {r['spans']} | {db(r['gn']):.2f} | {db(r['gn_incoherent']):.2f} | "
                + ("—" if r["egn"] is None else f"{db(r['egn']):.2f}")
                + f" | {db(r['first_order']):.2f} | {db(r['ssfm']):.2f} | {db(r['ssfm_plus3db']):.2f} |"
            )
    axes[0, 0].set_ylabel("η of the centre channel [dB(1/W²)]")
    axes[1, 0].set_ylabel("model − simulation [dB]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("3 × 32 GBd, 50 GHz grid, SSMF, ideal amplification", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "eta_vs_spans.png", dpi=150)
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results"))
