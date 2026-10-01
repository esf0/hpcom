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
    "qam+gauss": "QAM, CDC, Gaussian demapper",
    "qam+nn": "QAM, CDC, NN decoder (±3)",
    "gs+gauss": "learned points, CDC, Gaussian",
    "ae+mem": "NN encoder, CDC, NN decoder (±3)",
    "qam+dbp": "QAM, single-channel DBP, Gaussian",
    "gs+dbp": "learned points, single-channel DBP",
    "ae+mem+dbp": "NN encoder, single-channel DBP, NN decoder (±3)",
    "qam+ffdbp": "QAM, full-field DBP (ideal reference)",
}
STYLE = {"qam+ffdbp": {"ls": "--", "color": "0.4"}}


def main(folder: Path) -> None:
    data = json.loads((folder / "results.json").read_text())
    rows = data["results"]
    cfg = data["config"]
    methods = [m for m in LABELS if any(r["method"] == m for r in rows)]
    powers = sorted({r["power_dbm"] for r in rows})

    def get(method: str, power: float) -> dict:
        return next(r for r in rows if r["method"] == method and r["power_dbm"] == power)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, key, title in (
        (axes[0], "rates", f"training simulation ({cfg['steps_per_span']} steps/span)"),
        (axes[1], "rates_fine", f"4× finer simulation ({cfg['eval_steps_per_span']} steps/span)"),
    ):
        for m in methods:
            r = [np.asarray(get(m, p)[key]) for p in powers]
            ax.errorbar(
                powers,
                [x.mean() for x in r],
                yerr=[x.std() / np.sqrt(x.size) for x in r],
                marker="o",
                capsize=3,
                label=LABELS[m],
                **STYLE.get(m, {}),
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("launch power per channel [dBm]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rate of the centre channel [bit/symbol]")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"3 × 32 GBd, 50 GHz grid, M = {cfg['order']}, {cfg['spans']} × 80 km SSMF, NF 5 dB",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(folder / "rate_vs_power.png", dpi=150)

    lines = []
    for key, name in (("rates", "training simulation"), ("rates_fine", "4× finer simulation")):
        base = {p: np.asarray(get("qam+gauss", p)[key]) for p in powers}
        lines += [
            f"**{name}** — rate (paired gain over QAM + CDC + Gaussian ± s.e.)",
            "",
            "| method | " + " | ".join(f"{p:+.0f} dBm" for p in powers) + " | best |",
            "|---" * (len(powers) + 2) + "|",
        ]
        for m in methods:
            cells = []
            for p in powers:
                r = np.asarray(get(m, p)[key])
                d = r - base[p]
                if m == "qam+gauss":
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

    shaped = [m for m in methods if m in ("gs+gauss", "ae+mem", "gs+dbp", "ae+mem+dbp")]
    fig, axes = plt.subplots(1, len(shaped) + 1, figsize=(3.2 * (len(shaped) + 1), 3.3))
    qam = np.asarray(get("qam+gauss", powers[0])["points"]).reshape(-1, 2)
    axes[0].scatter(qam[:, 0], qam[:, 1], s=6)
    axes[0].set_title(f"{cfg['order']}-QAM", fontsize=9)
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
        np.mean(get("qam+gauss", p_hi)["rates"]), color="k", ls="--", lw=1, label="QAM + CDC (eval)"
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
