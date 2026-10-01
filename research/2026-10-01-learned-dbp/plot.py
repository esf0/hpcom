"""Figures and summary tables from ``results-*/results.json`` (see ``run.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "dbp4": "single-channel DBP, 4 steps/span (fixed)",
    "dbp1": "single-channel DBP, 1 step/span (fixed)",
    "ldbp1": "learned single-channel DBP, 1 step/span",
    "ldbp4": "learned single-channel DBP, 4 steps/span",
    "dbp4+gs+cnn": "DBP 4 (fixed), learned points, CNN decoder ±84",
    "ldbp4+gs+cnn": "learned DBP 4, learned points, CNN decoder ±84",
    "wdbp4": "wideband DBP of 3 channels, 4 steps/span (fixed)",
    "wdbp2": "wideband DBP of 3 channels, 2 steps/span (fixed)",
    "wdbp8": "wideband DBP of 3 channels, 8 steps/span (fixed)",
    "wdbp16": "wideband DBP of 3 channels, 16 steps/span (fixed)",
    "wldbp2": "learned wideband DBP, 2 steps/span",
    "ffdbp": "full-field DBP, forward steps (ideal reference)",
}
STYLE = {"ffdbp": {"ls": "--", "color": "0.4"}}


def main(folder: Path) -> None:
    data = json.loads((folder / "results.json").read_text())
    rows = data["results"]
    cfg = data["config"]
    methods = [m for m in LABELS if any(r["method"] == m for r in rows)]
    powers = sorted({r["power_dbm"] for r in rows})

    def get(method: str, power: float) -> dict:
        return next(r for r in rows if r["method"] == method and r["power_dbm"] == power)

    panels = (
        (
            "single-channel receivers",
            [
                ("dbp1", "C1", "--"),
                ("ldbp1", "C1", "-"),
                ("dbp4", "C0", "--"),
                ("ldbp4", "C0", "-"),
                ("dbp4+gs+cnn", "C2", "--"),
                ("ldbp4+gs+cnn", "C2", "-"),
            ],
        ),
        (
            "wideband (all three channels) vs full-field DBP",
            [
                ("wdbp2", "C3", "--"),
                ("wldbp2", "C3", "-"),
                ("wdbp8", "C4", "--"),
                ("wdbp16", "C5", "--"),
                ("dbp4", "C0", ":"),
                ("ffdbp", "0.3", "-."),
            ],
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (title, curves) in zip(axes, panels, strict=True):
        for m, color, ls in curves:
            if m not in methods:
                continue
            r = [np.asarray(get(m, p)["rates_fine"]) for p in powers]
            ax.errorbar(
                powers,
                [x.mean() for x in r],
                yerr=[x.std() / np.sqrt(x.size) for x in r],
                marker="o" if ls == "-" else "s",
                mfc=color if ls == "-" else "white",
                color=color,
                ls=ls,
                capsize=3,
                label=LABELS[m],
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("launch power per channel [dBm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("rate of the centre channel [bit/symbol]")
    fig.suptitle(
        f"3 × 32 GBd, 50 GHz grid, 256 points, {cfg['spans']} × 80 km SSMF, NF 5 dB — "
        f"evaluated on the 4× finer simulation; dashed: fixed, solid: learned",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(folder / "rate_vs_power.png", dpi=150)

    lines = []
    for key, name in (("rates", "training simulation"), ("rates_fine", "4× finer simulation")):
        base = {p: np.asarray(get("dbp4", p)[key]) for p in powers}
        lines += [
            f"**{name}** — rate (paired gain over fixed single-channel DBP ± s.e.)",
            "",
            "| method | " + " | ".join(f"{p:+.0f} dBm" for p in powers) + " | best |",
            "|---" * (len(powers) + 2) + "|",
        ]
        for m in methods:
            cells = []
            for p in powers:
                r = np.asarray(get(m, p)[key])
                d = r - base[p]
                if m == "dbp4":
                    cells.append(f"{r.mean():.3f}")
                else:
                    cells.append(
                        f"{r.mean():.3f} ({d.mean():+.3f}±{d.std() / np.sqrt(d.size):.3f})"
                    )
            best = max(np.mean(get(m, p)[key]) for p in powers)
            lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + f" | {best:.3f} |")
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    shaped = [m for m in methods if "gs" in m]
    fig, axes = plt.subplots(1, len(shaped) + 1, figsize=(3.2 * (len(shaped) + 1), 3.3))
    qam = np.asarray(get("dbp4", powers[0])["points"]).reshape(-1, 2)
    axes[0].scatter(qam[:, 0], qam[:, 1], s=6)
    axes[0].set_title("256-QAM", fontsize=9)
    for ax, m in zip(axes[1:], shaped, strict=True):
        p_best = max(powers, key=lambda p, m=m: np.mean(get(m, p)["rates_fine"]))
        pts = np.asarray(get(m, p_best)["points"]).reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color="C3")
        ax.set_title(f"{LABELS[m]}\n{p_best:+.0f} dBm", fontsize=7)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "constellations.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    p_hi = powers[-1]
    for m in methods:
        h = np.asarray(get(m, p_hi)["history"])
        if h.size:
            ax.plot(h[:, 0], h[:, 1], label=LABELS[m], alpha=0.8)
    ax.axhline(
        np.mean(get("dbp4", p_hi)["rates"]), color="k", ls="--", lw=1, label="QAM + CDC (eval)"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("batch rate [bit/symbol]")
    ax.set_title(f"training at {p_hi:+.0f} dBm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "training.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results-256"))
