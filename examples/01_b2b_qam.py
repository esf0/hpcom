"""01 — Back-to-back 16-QAM: BER vs. SNR against theory.

The smallest end-to-end hpcom scenario: a transmitter, an additive-noise
"link" that sets the per-symbol SNR Es/N0, and a matched-filter receiver.
The measured BER is printed next to the exact Gray-QAM formula.

Run::

    python examples/01_b2b_qam.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from hpcom import AdditiveNoise, Link, Simulation, SingleCarrierReceiver, SingleCarrierTransmitter


def ber_16qam_theory(snr_db: float) -> float:
    """Exact BER of Gray-coded 16-QAM on AWGN (Cho & Yoon, 2002)."""
    g = 10 ** (snr_db / 10)
    a = np.sqrt(g / 10)  # sqrt(3·SNR / (2(M−1))) for M = 16
    return float(0.75 * erfc(a) + 0.5 * erfc(3 * a) - 0.25 * erfc(5 * a)) / 2


def main() -> None:
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=1 << 16)
    rx = SingleCarrierReceiver(modulation_order=16, equalize="data_aided")

    print("16-QAM back-to-back with noise loading")
    print(f"{'SNR [dB]':>9} {'BER':>11} {'theory':>11} {'meas. SNR':>10} {'GMI [b/sym]':>12}")
    for snr_db in (8.0, 10.0, 12.0, 14.0, 16.0):
        noise = AdditiveNoise.for_snr(
            snr_db,
            power_per_polarization_w=tx.launch_power_per_polarization_w,
            symbol_rate_hz=tx.symbol_rate_hz,
        )
        result = Simulation(transmitter=tx, link=Link(elements=(noise,)), receiver=rx, seed=0).run()
        m = result.metrics
        print(
            f"{snr_db:9.1f} {m.ber:11.3e} {ber_16qam_theory(snr_db):11.3e} "
            f"{m.snr_db:10.2f} {m.gmi_bits:12.3f}"
        )


if __name__ == "__main__":
    main()
