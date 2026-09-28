"""09 — Digital back-propagation vs. linear dispersion compensation.

At high launch power the received signal is limited by self-phase modulation.
A receiver with ``compensation=DigitalBackPropagation(...)`` runs the link
equations backwards (noiselessly) on what its ADC delivers — the channel
band-limited and resampled to ``samples_per_symbol`` — and removes the
deterministic signal–signal nonlinearity, leaving ASE and signal–ASE
interaction. More steps per span help until the sampling rate limits: at 1
sample/symbol the roll-off band is already lost before DBP.

``Simulation.evaluate`` scores every receiver on the same received field
(same data, same noise), so the comparison is paired.

Run::

    python examples/09_digital_backpropagation.py
"""

from __future__ import annotations

from hpcom import (
    EDFA,
    DigitalBackPropagation,
    Fiber,
    Link,
    Simulation,
    SingleCarrierTransmitter,
)

BACKEND = "numpy"


def main() -> None:
    fiber = Fiber.ssmf(80.0, n_steps=100, polarization="dual", backend=BACKEND)
    link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=4.0))
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 12,
        polarization="dual",
        launch_power_per_channel_dbm=4.0,
        seed=1,
    )
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
        seed=2,
    )
    cdc = sim.run()
    print("16-QAM, 32 GBd, dual pol., 10 × 80 km, +4 dBm — every receiver sees the same field")
    print(f"  linear CD compensation       : SNR {cdc.metrics.snr_db:5.2f} dB")
    for sps in (1, 2, 4):
        for steps in (1, 4, 16):
            dbp = DigitalBackPropagation(link=link, steps_per_span=steps, samples_per_symbol=sps)
            result = sim.evaluate(cdc, tx.matched_receiver(compensation=dbp))
            print(
                f"  DBP {sps} sample(s)/symbol, {steps:2d} steps/span : "
                f"SNR {result.metrics.snr_db:5.2f} dB"
            )


if __name__ == "__main__":
    main()
