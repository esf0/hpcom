"""02 — Dual-polarisation 16-QAM over 10 × 80 km SSMF with EDFAs: launch-power sweep.

The flagship scenario. For each launch power the signal is propagated with
the Manakov equation, dispersion is compensated in the receiver and the mean
nonlinear phase rotation removed by data-aided equalisation. Low powers are
ASE-limited, high powers nonlinearity-limited: the SNR peaks at the optimum
launch power.

Switch ``BACKEND`` to ``"tensorflow"`` (``pip install "hpcom[tf]"``) to run on GPU.

Run::

    python examples/02_ssmf_edfa_chain.py
"""

from __future__ import annotations

import numpy as np

from hpcom import EDFA, Fiber, Link, Simulation, SingleCarrierReceiver, SingleCarrierTransmitter

BACKEND = "numpy"


def main() -> None:
    fiber = Fiber.ssmf(length_km=80.0, n_steps=40, polarization="dual", backend=BACKEND)
    edfa = EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0)
    link = Link.from_spans(n_spans=10, fiber=fiber, amplifier=edfa)

    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 12,
        samples_per_symbol=4,
        polarization="dual",
    )
    rx = SingleCarrierReceiver(
        modulation_order=16,
        samples_per_symbol=4,
        polarization="dual",
        equalize="data_aided",
        cd_compensation_s2=link.total_dispersion_s2,
    )

    print(
        f"DP-16QAM, 32 GBd, {link.n_spans} × {fiber.length_km:.0f} km SSMF, EDFA NF 5 dB ({BACKEND})"
    )
    print(f"{'P_ch [dBm]':>11} {'SNR [dB]':>9} {'BER':>11} {'GMI [b/sym]':>12} {'NL phase':>9}")
    for power_dbm in (-4.0, -2.0, 0.0, 2.0, 4.0):
        sim = Simulation(
            transmitter=tx.with_launch_power(per_channel_dbm=power_dbm),
            link=link,
            receiver=rx,
            seed=0,
        )
        result = sim.run()
        m = result.metrics
        phase = np.degrees(np.angle(result.received.gain)).mean()
        print(f"{power_dbm:11.1f} {m.snr_db:9.2f} {m.ber:11.3e} {m.gmi_bits:12.3f} {phase:8.1f}°")
    print("\nLast run:", result)


if __name__ == "__main__":
    main()
