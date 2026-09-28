"""04 — Dataset generation for ML / data-mining workflows.

Runs one scenario over many seeds and launch powers. Each realisation is one
``.npz`` file (``Result.save``) holding the transmitted bits/symbols/field,
the received field and symbols, the metrics and the full configuration;
``Result.load`` restores it, and ``np.load`` reads the arrays directly
(``transmitted.symbols``, ``received.symbols``, ``received_field``, …).
``index.json`` lists every file with its metrics.

Run::

    python examples/04_dataset_generation.py [output-directory]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from hpcom import EDFA, Fiber, Link, Simulation, SingleCarrierReceiver, SingleCarrierTransmitter


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fiber = Fiber.ssmf(length_km=80.0, n_steps=20)
    link = Link.from_spans(
        n_spans=5, fiber=fiber, amplifier=EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0)
    )
    base = Simulation(
        transmitter=SingleCarrierTransmitter(
            modulation_order=16, symbol_rate_hz=32e9, n_symbols=2048, samples_per_symbol=4
        ),
        link=link,
        receiver=SingleCarrierReceiver(
            modulation_order=16,
            samples_per_symbol=4,
            equalize="data_aided",
            cd_compensation_s2=link.total_dispersion_s2,
        ),
    )
    (out_dir / "config.json").write_text(base.model_dump_json(indent=2))

    index = []
    for power_dbm in (-2.0, 0.0, 2.0):
        for seed in range(3):
            sim = base.replace(
                transmitter=base.transmitter.with_launch_power(per_channel_dbm=power_dbm), seed=seed
            )
            result = sim.run()
            name = f"p{power_dbm:+.0f}dBm_seed{seed}.npz"
            result.save(out_dir / name)  # arrays + full config; Result.load() restores it
            index.append(
                {
                    "file": name,
                    "launch_power_per_channel_dbm": power_dbm,
                    **result.to_dict()["metrics"],
                    "seed": seed,
                }
            )
            print(f"  {name}: BER={result.metrics.ber:.2e}, SNR={result.metrics.snr_db:.2f} dB")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index)} realisations to {out_dir}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="hpcom_dataset_"))
    )
