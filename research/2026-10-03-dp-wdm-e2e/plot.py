"""Rate vs power and a summary table from ``results/results.json`` (see ``run.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "qam+cdc": ("QAM, CDC (fixed)", "C0", "--"),
    "gs+cdc": ("learned constellation, CDC", "C0", "-"),
    "gs+txcnn+cdc": ("learned constellation + Tx CNN pre-distorter, CDC", "C2", "-"),
    "gs+wdbp2": ("learned constellation, wideband DBP 2 steps/span (fixed)", "C3", "--"),
    "gs+wldbp2": ("learned constellation, learned wideband DBP 2 steps/span", "C3", "-"),
    "gs+txcnn+wldbp2": ("constellation + Tx CNN + learned wideband DBP, jointly", "C1", "-"),
    "ffdbp": ("QAM, full-field DBP, forward steps (ideal)", "0.3", "-."),
}


def main(folder: Path) -> None:
    data = json.loads((folder / "results.json").read_text())
    rows = data["results"]
    methods = [m for m in LABELS if any(r["method"] == m for r in rows)]
    powers = sorted({r["power_dbm"] for r in rows})

    def get(method: str, power: float) -> dict:
        return next(r for r in rows if r["method"] == method and r["power_dbm"] == power)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in methods:
        label, color, ls = LABELS[m]
        r = [np.asarray(get(m, p)["rates_fine"]) for p in powers]
        ax.errorbar(
            powers,
            [x.mean() for x in r],
            yerr=[x.std() / np.sqrt(x.size) for x in r],
            color=color,
            ls=ls,
            marker="o" if ls == "-" else "s",
            mfc=color if ls == "-" else "white",
            capsize=3,
            label=label,
        )
    ax.set_xlabel("launch power per channel (both polarisations) [dBm]")
    ax.set_ylabel("rate of the centre channel [bit/symbol/pol]")
    ax.set_title(
        "DP 5 × 32 GBd, 50 GHz grid, 64 points, 10 × 80 km SSMF, NF 5 dB — finer simulation",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(folder / "rate_vs_power.png", dpi=150)

    lines = []
    for key, name in (("rates", "training simulation"), ("rates_fine", "finer simulation")):
        base = {p: np.asarray(get("qam+cdc", p)[key]) for p in powers}
        lines += [
            f"**{name}** — bit/symbol/pol (paired gain over QAM + CDC ± s.e.; ×2 for bit/4D)",
            "",
            "| method | " + " | ".join(f"{p:+.0f} dBm" for p in powers) + " | best |",
            "|---" * (len(powers) + 2) + "|",
        ]
        for m in methods:
            cells = []
            for p in powers:
                r = np.asarray(get(m, p)[key])
                d = r - base[p]
                cells.append(
                    f"{r.mean():.3f}"
                    if m == "qam+cdc"
                    else f"{r.mean():.3f} ({d.mean():+.3f}±{d.std() / np.sqrt(d.size):.3f})"
                )
            best = max(np.mean(get(m, p)[key]) for p in powers)
            lines.append(f"| {LABELS[m][0]} | " + " | ".join(cells) + f" | {best:.3f} |")
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results"))
