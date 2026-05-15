"""04 — Bulk dataset generation for ML / data-mining workflows.

Loops a single scenario across many random seeds and writes the (sent, received)
symbol pairs to ``.npz`` files. This is the primary on-ramp for the strategic
ML/data-mining audience — once you have a scenario you trust, this script
turns it into a labelled corpus for training nonlinearity-compensation,
equalization, or denoising models.

Run::

    python examples/04_dataset_generation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hpcom import EDFA, Fiber, Link, Receiver, Simulation, Transmitter


def main(out_dir: str = "_dataset", n_realizations: int = 4) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    fiber = Fiber.ssmf(length_km=80.0, n_steps=20, backend="tensorflow")
    edfa = EDFA(gain_db=fiber.alpha_db_per_km * fiber.length_km, noise_figure_db=5.0)
    link = Link.from_spans(n_spans=3, fiber=fiber, amplifier=edfa)

    rx = Receiver(
        modulation_order=16,
        samples_per_symbol=4,
        roll_off=0.1,
        pulse_span_symbols=16,
        equalize="unit_power",
        cd_compensation_s2=link.total_dispersion_s2,
    )

    for seed in range(n_realizations):
        tx = Transmitter(
            modulation_order=16,
            symbol_rate_hz=32e9,
            n_symbols=4096,
            samples_per_symbol=4,
            roll_off=0.1,
            pulse_span_symbols=16,
            seed=seed,
        )
        sent = tx.generate()
        result = Simulation(
            transmitter=tx,
            link=link,
            receiver=rx,
            seed=seed,
            launch_power_dbm=-3.0,
        ).run()

        path = out_path / f"realization_{seed:04d}.npz"
        np.savez_compressed(
            path,
            sent_symbols=sent.symbols,
            sent_bits=sent.bits,
            decoded_symbols=result.decoded_symbols,
            decoded_bits=result.decoded_bits,
            ber=result.metrics.ber,
            evm_rms=result.metrics.evm_rms,
        )
        print(f"  wrote {path}  (BER={result.metrics.ber:.2e})")

    print(f"Done. {n_realizations} realisations under {out_path}/")


if __name__ == "__main__":
    main()
