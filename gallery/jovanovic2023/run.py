"""Jovanovic, Da Ros, Zibar, Yankov (JLT 41(12), 2023), Fig. 5: geometric shaping learned through the SSFM.

Dual-polarisation 5 × 32 GBd WDM, 50 GHz grid, RRC 0.01, 64 points,
10 × 100 km SSMF (D = 16.464 ps/(nm·km), γ = 1.3 /W/km), EDFA NF 5 dB;
mismatched-Gaussian-receiver MI of the centre channel. The paper trains
the constellation by back-propagation through the SSFM (+0.21 bit/4D over
64-QAM at 0.5 dBm). hpcom: the ``jovanovic2023`` benchmark scenario,
methods ``qam+cdc`` and ``gs+cdc`` (the Gaussian-receiver MI is the
training objective), scored on a 4× finer simulation.

Run (GPU)::

    python run.py --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpcom import bench, learn

#: Read off the paper's Fig. 5, bit/4D-symbol (about ±0.01).
PAPER = {
    "qam+cdc": {
        -1.5: 10.37,
        -1.0: 10.52,
        -0.5: 10.63,
        0.0: 10.69,
        0.5: 10.71,
        1.0: 10.66,
        1.5: 10.57,
        2.0: 10.39,
    },
    "gs+cdc": {
        -1.5: 10.59,
        -1.0: 10.73,
        -0.5: 10.83,
        0.0: 10.90,
        0.5: 10.92,
        1.0: 10.88,
        1.5: 10.78,
        2.0: 10.62,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--powers", type=float, nargs="+", default=[-1.5, -0.5, 0.5, 1.5])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenario = bench.scenario("jovanovic2023")
    if args.quick:
        scenario = scenario.model_copy(
            update={
                "transmitter": scenario.transmitter.replace(n_symbols=320),
                "spans": 2,
                "steps_per_span": 6,
                "evaluation_steps_per_span": 12,
                "evaluation_blocks": 2,
            }
        )
        args.powers, args.steps = [0.5], 2
    results = {}
    for name in ("qam+cdc", "gs+cdc"):
        r = bench.run(
            scenario, bench.METHODS[name], learn.Trainer(steps=args.steps), powers_dbm=args.powers
        )
        results[name] = r
        for row in r.rows:
            print(
                f"{name} {row.power_dbm:+.1f} dBm  {2 * row.rate_scored:.3f} bit/4D  (paper {PAPER[name].get(row.power_dbm)})",
                flush=True,
            )
    (args.out / "results.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in results.values()], indent=1)
    )
    lines = [
        "| power | paper QAM | hpcom QAM | paper learned | hpcom learned | paper gain | hpcom gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for qam, gs in zip(results["qam+cdc"].rows, results["gs+cdc"].rows, strict=True):
        p = qam.power_dbm
        pq, pg = PAPER["qam+cdc"].get(p), PAPER["gs+cdc"].get(p)
        paper_gain = f"{pg - pq:+.2f}" if pq is not None and pg is not None else "—"
        lines.append(
            f"| {p:+.1f} dBm | {pq if pq is not None else '—'} | {2 * qam.rate_scored:.2f} | "
            f"{pg if pg is not None else '—'} | {2 * gs.rate_scored:.2f} | {paper_gain} | "
            f"{2 * (gs.rate_scored - qam.rate_scored):+.2f} |"
        )
    (args.out / "summary.md").write_text(
        "MI of the centre channel, bit/4D-symbol\n\n" + "\n".join(lines) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
