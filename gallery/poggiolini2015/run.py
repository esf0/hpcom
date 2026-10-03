"""Poggiolini, Jiang, Carena, Forghieri (arXiv:1503.04132, 2015), Figs. 1(a) and 2(a).

NLI coefficient η of the centre channel vs number of spans: PM-QPSK,
32 GBd, roll-off 0.05, 33.6 GHz grid, 3 or 15 channels, 100 km SMF spans
(D = 16.7 ps/(nm·km), γ = 1.3 /W/km, 0.22 dB/km), ideal noiseless
amplification. hpcom: the GN model and the closed-form EGN correction
(``hpcom.nli``), and the first-order perturbation of the actual signal,
which equals the split-step simulation within 0.05 dB
(``research/2026-10-01-gn-egn``), as the counterpart of the paper's
simulation curve (3 channels).

Run::

    python run.py --out results        # a few minutes on a CPU
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hpcom import EDFA, Fiber, Link, WDMConfig, WDMTransmitter, nli

#: Read off the paper's figures, dB(1/W²), about ±0.2 dB.
PAPER = {
    "3ch": {
        "GN": {1: 26.7, 2: 30.0, 5: 34.3, 10: 37.6, 20: 40.9, 50: 45.3},
        "closed-form EGN": {1: 24.5, 2: 27.9, 5: 32.5, 10: 36.0, 20: 39.4, 50: 43.9},
        "simulation": {1: 21.2, 2: 26.8, 5: 32.2, 10: 35.9, 20: 39.4, 50: 44.2},
    },
    "15ch": {
        "GN": {1: 29.5, 2: 32.7, 5: 36.9, 10: 40.1, 20: 43.3, 50: 47.5},
        "closed-form EGN": {1: 27.5, 2: 30.8, 5: 35.1, 10: 38.4, 20: 41.6, 50: 45.9},
        "simulation": {1: 23.6, 50: 46.2},
    },
}


def db(x: float) -> float:
    return 10.0 * math.log10(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spans_list = [1, 2] if args.quick else [1, 2, 5, 10, 20, 50]
    fiber = Fiber.ssmf(100.0, dispersion_ps_per_nm_per_km=16.7, polarization="dual").replace(
        alpha_db_per_km=0.22, gamma_per_w_per_km=1.3
    )
    rows = []
    for system, n_ch in (("3ch", 3), ("15ch", 15)):
        tx = WDMTransmitter(
            modulation_order=4,
            symbol_rate_hz=32e9,
            roll_off=0.05,
            n_symbols=240 if args.quick else 1000,
            polarization="dual",
            samples_per_symbol=2 * n_ch,
            wdm=WDMConfig(n_channels=n_ch, spacing_hz=33.6e9),
        )
        for spans in spans_list:
            link = Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
            ours = {
                "GN": db(nli.nli_coefficient(link, tx)),
                "closed-form EGN": db(nli.nli_coefficient(link, tx, model="egn")),
            }
            if n_ch == 3:  # the simulation's counterpart: first order on the actual signal
                first = nli.first_order_nli(
                    link, tx, n_blocks=1 if args.quick else 4, nodes_per_fiber=96
                )
                ours["simulation"] = db(first.eta_per_w2)
            for quantity, value in ours.items():
                paper = PAPER[system][quantity].get(spans)
                rows.append(
                    {
                        "system": system,
                        "spans": spans,
                        "quantity": quantity,
                        "paper": paper,
                        "hpcom": value,
                    }
                )
                print(
                    f"{system} {spans:3d} spans  {quantity:16s} paper {paper}  hpcom {value:.2f}",
                    flush=True,
                )
    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    lines = ["| system | quantity | spans | paper | hpcom | Δ |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["paper"] is not None:
            lines.append(
                f"| {r['system']} | {r['quantity']} | {r['spans']} | {r['paper']:.1f} | "
                f"{r['hpcom']:.2f} | {r['hpcom'] - r['paper']:+.2f} |"
            )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
