"""Häger & Pfister (IEEE JSAC 39(1), 2021; arXiv:2010.14258), Fig. 5: learned DBP at one step per span.

Single polarisation, 10.7 GBd, Gaussian symbols, RRC 0.1, 25 × 80 km SSMF
(β₂ = −21.683 ps²/km, γ = 1.3 /W/km, 0.2 dB/km), EDFA NF 5 dB; receiver at
2 samples/symbol; effective SNR. The paper compares linear equalisation,
DBP with one step per span (frequency domain) and its learned DBP with
77 FIR taps in total. hpcom: CD compensation, ``DBP(steps_per_span=1)``
(effective-length Kerr step at mid-span, the default) and
``LearnedDBP(steps_per_span=1)`` (every step's frequency response, Kerr
coefficient and power filter trained; not tap-constrained), trained on the
effective SNR through the split-step simulation and scored on a 4× finer
one.

Run (GPU)::

    python run.py --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hpcom import EDFA, ConstellationConfig, Fiber, Link, SingleCarrierTransmitter, learn

#: Read off the paper's Fig. 5, effective SNR in dB (about ±0.2 dB).
PAPER = {
    "linear equalisation": {-10.0: 16.4, -8.0: 17.3, -7.0: 17.4, -6.0: 16.9},
    "DBP 1 step/span": {-6.0: 20.6, -4.0: 22.6, -2.0: 24.2, 0.0: 25.3, 1.0: 25.1, 2.0: 23.6},
    "learned DBP 1 step/span (77 taps)": {-4.0: 22.1, -2.0: 23.3, -1.0: 23.4, 0.0: 22.8, 2.0: 17.6},
}


def gaussian_constellation(m: int = 1024, seed: int = 0) -> ConstellationConfig:
    """``m`` points drawn from a circular Gaussian, sent uniformly: Gaussian-like symbols."""
    rng = np.random.default_rng(seed)
    return ConstellationConfig.from_points(rng.standard_normal(m) + 1j * rng.standard_normal(m))


def link_with(steps_per_span: int, spans: int) -> Link:
    fiber = Fiber(
        length_km=80.0,
        alpha_db_per_km=0.2,
        beta2_s2_per_m=-21.683e-27,
        gamma_per_w_per_km=1.3,
        n_steps=steps_per_span,
    ).with_steps(distribution="logarithmic", log_factor=1 / 3)
    return Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--powers", type=float, nargs="+", default=[-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0]
    )
    parser.add_argument("--learned-powers", type=float, nargs="+", default=[-4.0, -2.0, 0.0, 2.0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spans, train_steps, score_steps, n_symbols, blocks = 25, 100, 400, 1024, 16
    if args.quick:
        spans, train_steps, score_steps, n_symbols, blocks = 2, 4, 8, 256, 2
        args.powers, args.learned_powers, args.steps = [0.0], [0.0], 2
    link, scoring = link_with(train_steps, spans), link_with(score_steps, spans)
    methods = {
        "linear equalisation": learn.CDC(),
        "DBP 1 step/span": learn.DBP(steps_per_span=1, bandwidth_factor=2.0),
        "learned DBP 1 step/span": learn.LearnedDBP(steps_per_span=1, bandwidth_factor=2.0),
    }
    rows = []
    for name, front_end in methods.items():
        powers = args.learned_powers if isinstance(front_end, learn.LearnedDBP) else args.powers
        for p in powers:
            tx = SingleCarrierTransmitter(
                constellation=gaussian_constellation(),
                symbol_rate_hz=10.7e9,
                n_symbols=n_symbols,
                samples_per_symbol=6,
                launch_power_per_channel_dbm=p,
            )
            system = learn.EndToEnd(
                transmitter=tx, link=link, front_end=front_end, decoder=learn.EffectiveSNR()
            )
            trained = learn.Trainer(steps=args.steps).fit(system)
            seed = 1_000_000 + round(10 * p)
            snr = learn.evaluate(system, trained.params, blocks=blocks, seed=seed, link=scoring)
            rows.append(
                {
                    "method": name,
                    "power_dbm": p,
                    "snr_db": snr.mean,
                    "stderr": snr.stderr,
                    "train_s": trained.seconds,
                }
            )
            print(f"{name:26s} {p:+5.1f} dBm  {snr.mean:.2f} ± {snr.stderr:.2f} dB", flush=True)
    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| method | paper peak | hpcom peak | hpcom at the paper's peak power |",
        "|---|---|---|---|",
    ]
    for name, paper in PAPER.items():
        ours = [r for r in rows if r["method"] == name.replace(" (77 taps)", "")]
        if not ours:
            continue
        p_peak = max(paper, key=paper.get)
        best = max(ours, key=lambda r: r["snr_db"])
        at = next((r for r in ours if r["power_dbm"] == p_peak), None)
        at_text = "—" if at is None else f"{at['snr_db']:.2f} dB"
        lines.append(
            f"| {name} | {paper[p_peak]:.1f} dB at {p_peak:+.0f} dBm | {best['snr_db']:.2f} dB at "
            f"{best['power_dbm']:+.0f} dBm | {at_text} |"
        )
    (args.out / "summary.md").write_text("Effective SNR\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
