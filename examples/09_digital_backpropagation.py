"""09 — Digital back-propagation vs. linear dispersion compensation.

At high launch power the received signal is limited by self-phase modulation.
``Link.backpropagate`` runs the link equations backwards (noiselessly); with
enough steps per span it removes the deterministic signal–signal
nonlinearity, leaving ASE and signal–ASE interaction.

Run::

    python examples/09_digital_backpropagation.py
"""

from __future__ import annotations

import numpy as np

from hpcom import EDFA, Fiber, Link, SingleCarrierReceiver, SingleCarrierTransmitter
from hpcom.metrics import snr_db

BACKEND = "numpy"


def main() -> None:
    fiber = Fiber.ssmf(80.0, n_steps=100, backend=BACKEND)
    link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=4.0))
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 12,
        samples_per_symbol=4,
        launch_power_per_channel_dbm=4.0,
        seed=1,
    )
    cdc_rx = SingleCarrierReceiver(
        modulation_order=16,
        samples_per_symbol=4,
        equalize="data_aided",
        cd_compensation_s2=link.total_dispersion_s2,
    )
    dbp_rx = cdc_rx.replace(cd_compensation_s2=0.0)  # the back-propagation removes the dispersion

    sent = tx.generate()
    out = link.propagate(sent.field, sent.dt_s, rng=np.random.default_rng(2))
    print("16-QAM, 32 GBd, 10 × 80 km, +4 dBm")
    cdc = cdc_rx.receive(out, sent.dt_s, reference=sent)
    print(f"  linear CD compensation : SNR {snr_db(cdc.symbols, sent.data_symbols()):5.2f} dB")
    for steps in (1, 5, 20):
        field = link.backpropagate(out, sent.dt_s, steps_per_span=steps)
        dbp = dbp_rx.receive(field, sent.dt_s, reference=sent)
        print(
            f"  DBP, {steps:2d} steps/span     : SNR {snr_db(dbp.symbols, sent.data_symbols()):5.2f} dB"
        )


if __name__ == "__main__":
    main()
