"""05 — A custom waveform through the full pipeline.

:mod:`hpcom.experimental` holds a complete, minimal transmitter/receiver
pair — rectangular-pulse QAM — written against the public extension
contract. Copy that module, rename the classes, and replace the blocks marked
``>>> REPLACE`` with your waveform; everything else (links, polarisation,
launch power, Simulation, metrics) works unchanged.

Run::

    python examples/05_custom_waveform.py
"""

from __future__ import annotations

from hpcom import EDFA, Fiber, Link, Simulation
from hpcom.experimental import ExperimentalReceiver, ExperimentalTransmission


def main() -> None:
    fiber = Fiber.ssmf(80.0, n_steps=20, polarization="dual")
    link = Link.from_spans(3, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = ExperimentalTransmission(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=4096,
        polarization="dual",
        launch_power_per_channel_dbm=-2.0,
    )
    rx = ExperimentalReceiver(
        modulation_order=16,
        polarization="dual",
        equalize="data_aided",
        cd_compensation_s2=link.total_dispersion_s2,
    )
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
    print("Rectangular-pulse DP-16QAM, 10 GBd, 3 × 80 km SSMF")
    print(result)
    for stream in result.stream_metrics:
        print(f"  pol {stream.polarization}: SNR {stream.snr_db:.2f} dB, BER {stream.ber:.2e}")


if __name__ == "__main__":
    main()
