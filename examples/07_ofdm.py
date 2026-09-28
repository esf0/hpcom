"""07 — CP-OFDM over a short dispersive link with pilot-based equalisation.

The cyclic prefix absorbs the dispersion of a short link, so a one-tap
per-subcarrier equaliser driven by the pilot subcarriers is enough — no
separate dispersion compensation. The IFFT is 2× oversampled so the SSFM has
room for nonlinear spectral broadening.

Run::

    python examples/07_ofdm.py
"""

from __future__ import annotations

from hpcom import EDFA, Fiber, Link, OFDMConfig, OFDMReceiver, OFDMTransmitter, Simulation


def main() -> None:
    ofdm = OFDMConfig(
        n_subcarriers=256,
        cp_len=32,
        n_guard=8,
        pilot_subcarrier_indices=tuple(k for k in range(0, 256, 8) if not 120 <= k < 136),
        oversampling=2,
    )
    tx = OFDMTransmitter(
        modulation_order=16,
        subcarrier_spacing_hz=100e6,
        n_frames=64,
        ofdm=ofdm,
        polarization="dual",
        launch_power_per_channel_dbm=-2.0,
    )
    rx = OFDMReceiver(modulation_order=16, ofdm=ofdm, polarization="dual", equalize="pilot")
    fiber = Fiber.ssmf(20.0, n_steps=20, polarization="dual")
    link = Link(elements=(fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0)))
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
    print(
        f"DP-16QAM CP-OFDM: {ofdm.n_data} data + {ofdm.n_pilots} pilot subcarriers, "
        f"{tx.bandwidth_hz / 1e9:.1f} GHz, 20 km SSMF"
    )
    print(result)


if __name__ == "__main__":
    main()
