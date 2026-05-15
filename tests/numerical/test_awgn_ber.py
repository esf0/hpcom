"""Measured BER vs. theoretical AWGN curve for back-to-back QAM.

For Gray-coded square M-QAM with coherent detection over an AWGN channel,
the bit error probability is well approximated (high SNR) by

    P_b ≈ (4 / k) · (1 − 1/√M) · Q( sqrt(3 · SNR / (M − 1)) )

with k = log₂ M and Q(x) = (1/2) · erfc(x / √2). SNR is the per-symbol
signal-to-noise ratio at the post-matched-filter symbol decision point.

Reference: Proakis & Salehi, *Digital Communications* (5th ed.), §4.4.

This test injects complex AWGN onto the field between the transmitter and
the receiver matched filter, then compares the measured BER against the
theoretical curve. The receiver runs with ``equalize="none"``, so any
pulse-shaping or matched-filter mis-calibration would surface as a BER
offset rather than being silently absorbed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from hpcom.modulation import bits_per_symbol
from hpcom.receiver import Receiver
from hpcom.transmitter import Transmitter


def _q(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def _theoretical_ber_qam(snr_linear: float, m: int) -> float:
    k = bits_per_symbol(m)
    return float((4.0 / k) * (1.0 - 1.0 / np.sqrt(m)) * _q(np.sqrt(3.0 * snr_linear / (m - 1))))


def _run_b2b_with_awgn(modulation_order: int, snr_db: float, n_symbols: int, seed: int) -> float:
    sps = 8
    tx = Transmitter(
        modulation_order=modulation_order,
        symbol_rate_hz=32e9,
        n_symbols=n_symbols,
        samples_per_symbol=sps,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=seed,
    )
    sent = tx.generate()

    # Inject white complex Gaussian on the field. The matched filter has unit
    # energy (Σh² = 1) so per-symbol post-MF noise variance equals σ²_field —
    # i.e. SNR_per_symbol = E[|s|²] / σ²_field = 1 / σ²_field for our
    # unit-power constellation.
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2_field = 1.0 / snr_lin
    rng = np.random.default_rng(seed + 1)
    noise = (rng.standard_normal(sent.field.size) + 1j * rng.standard_normal(sent.field.size))
    noise *= np.sqrt(sigma2_field / 2.0)
    rx_field = sent.field + noise

    rx = Receiver(
        modulation_order=modulation_order,
        samples_per_symbol=sps,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        equalize="none",
    )
    received = rx.receive(rx_field)

    skip = tx.pulse_span_symbols
    bps = bits_per_symbol(modulation_order)
    sent_bits = sent.bits.reshape(-1, bps)[skip:-skip].ravel()
    rx_bits = received.bits.reshape(-1, bps)[skip:-skip].ravel()
    return float(np.mean(sent_bits != rx_bits))


@pytest.mark.parametrize(
    "order, snr_db, tol_db",
    [
        (4, 7.0, 0.5),
        (4, 10.0, 0.7),
        (16, 14.0, 0.7),
        (16, 17.0, 0.8),
    ],
)
def test_measured_ber_within_tolerance_of_theory(order: int, snr_db: float, tol_db: float):
    """BER measured at a given SNR must lie within ``tol_db`` of theory.

    "Within tol_db of theory" means the SNR that would explain the measured
    BER on the theoretical curve differs from the actual SNR by < tol_db.
    Tighter than ~0.5 dB is dominated by Monte-Carlo noise at these symbol
    counts; we accept that.
    """
    measured = _run_b2b_with_awgn(order, snr_db, n_symbols=20_000, seed=0)
    theoretical = _theoretical_ber_qam(10 ** (snr_db / 10.0), order)
    # SNR-domain tolerance: invert the Q-curve numerically by sweeping.
    snr_grid = np.arange(snr_db - 3.0, snr_db + 3.0, 0.05)
    ber_grid = np.array([_theoretical_ber_qam(10 ** (s / 10.0), order) for s in snr_grid])
    # Equivalent SNR for measured BER (clamp away from zero to avoid log issues).
    target = max(measured, 1e-7)
    eq_snr_db = float(snr_grid[int(np.argmin(np.abs(ber_grid - target)))])
    delta_db = abs(eq_snr_db - snr_db)
    assert delta_db < tol_db, (
        f"BER {measured:.2e} at SNR {snr_db} dB (order {order}); "
        f"theoretical {theoretical:.2e}; equiv SNR {eq_snr_db:.2f} dB; Δ = {delta_db:.2f} dB"
    )
