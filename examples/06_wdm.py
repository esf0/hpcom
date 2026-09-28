"""06 — Five-channel dual-polarisation WDM: per-channel performance.

Five 32 GBd channels on a 50 GHz grid share the fiber. Nonlinear crosstalk
(XPM/FWM) is strongest for the centre channel; the receiver demultiplexes
every channel and ``Result.stream_metrics`` reports each (polarisation,
channel) separately.

Run::

    python examples/06_wdm.py
"""

from __future__ import annotations

from hpcom import EDFA, Fiber, Link, Simulation, WDMConfig, WDMReceiver, WDMTransmitter

BACKEND = "numpy"


def main() -> None:
    grid = WDMConfig(n_channels=5, spacing_hz=50e9)
    fiber = Fiber.ssmf(80.0, n_steps=80, polarization="dual", backend=BACKEND)
    link = Link.from_spans(5, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 12,
        samples_per_symbol=12,
        wdm=grid,
        polarization="dual",
        launch_power_per_channel_dbm=1.0,
    )
    rx = WDMReceiver(
        modulation_order=16,
        samples_per_symbol=12,
        wdm=grid,
        polarization="dual",
        equalize="data_aided",
        cd_compensation_s2=link.total_dispersion_s2,
    )
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
    print(
        f"5 × 32 GBd DP-16QAM, 50 GHz grid, {link.n_spans} × 80 km, "
        f"{tx.launch_power_total_w * 1e3:.1f} mW total launch power"
    )
    print(result)
    for ch in range(grid.n_channels):
        x, y = result.stream(polarization=0, channel=ch), result.stream(polarization=1, channel=ch)
        f_ghz = grid.channel_frequencies_hz()[ch] / 1e9
        print(
            f"  channel {ch} ({f_ghz:+6.0f} GHz): SNR {x.snr_db:5.2f} / {y.snr_db:5.2f} dB (X / Y)"
        )


if __name__ == "__main__":
    main()
