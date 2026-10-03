"""σ_max of the link Jacobian vs launch power and number of spans (see ``run.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(folder: Path) -> None:
    rows = json.loads((folder / "results.json").read_text())["rows"]
    systems = sorted({r["system"] for r in rows})
    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    lines = [
        "| system | spans | "
        + " | ".join(f"{p:+.0f} dBm" for p in sorted({r["power_dbm"] for r in rows}))
        + " |"
    ]
    lines.append("|---" * (2 + len({r["power_dbm"] for r in rows})) + "|")
    for ax, system in zip(axes, systems, strict=True):
        for spans in sorted({r["spans"] for r in rows if r["system"] == system}):
            sel = sorted(
                (r for r in rows if r["system"] == system and r["spans"] == spans),
                key=lambda r: r["power_dbm"],
            )
            p = [r["power_dbm"] for r in sel]
            gain_db = [20 * np.log10(r["singular_values"][0]) for r in sel]
            ax.plot(p, gain_db, marker="o", label=f"{spans} spans")
            lines.append(
                f"| {system} | {spans} | " + " | ".join(f"{g:.2f}" for g in gain_db) + " |"
            )
        ax.set_title({"1ch": "single channel", "3ch": "3 channels, 50 GHz"}.get(system, system))
        ax.set_xlabel("launch power per channel [dBm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("largest gradient gain 20·log₁₀ σ_max [dB]  (smallest = −same)")
    fig.suptitle(
        "Jacobian of a transparent SSMF link, 32 GBd dual-pol 16-QAM, 80 km spans", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(folder / "sigma_max.png", dpi=150)
    (folder / "summary.md").write_text("σ_max in dB (20·log₁₀)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results"))
