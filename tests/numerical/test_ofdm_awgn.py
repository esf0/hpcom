"""T12.4: OFDM AWGN BER curve — measured vs theoretical.

Adds complex AWGN to an OFDM field and verifies that the measured BER
matches the theoretical QAM-AWGN BER for the per-subcarrier SNR at
three operating points around BER ≈ 1e-2 to 1e-3.

Physics:
    The Tx normalises so that the data-bearing part of the field has unit
    average sample power. AWGN with variance ``σ²`` per complex sample
    spreads uniformly over all ``N`` subcarriers, but only ``n_data``
    carry data. After FFT and power de-normalisation, the per-data-
    subcarrier noise power is ``σ² · n_data / N`` (Parseval scaled by
    ``sqrt(n_data)/N``), and the signal power is 1. So the effective
    per-subcarrier SNR is::

        SNR_sub = 1 / (σ² · n_data / N) = N / (n_data · σ²)

    or equivalently::

        σ² = N / (n_data · SNR_sub)

    The theoretical BER for M-QAM over AWGN at per-symbol ``Es/N0``
    (which equals ``SNR_sub`` when symbols have unit power) is used as
    the reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc  # type: ignore[import-untyped]

from hpcom.modulation import bits_per_symbol
from hpcom.ofdm import OFDMConfig, OFDMReceiver, OFDMTransmitter

# ---------------------------------------------------------------------------
# Theoretical BER for M-QAM over AWGN
# ---------------------------------------------------------------------------


def _qam_ber_theory(m: int, es_n0_linear: float) -> float:
    """Approximate BER for square M-QAM over AWGN at per-symbol Es/N0.

    Uses the standard union-bound approximation (exact for QPSK,
    tight for M ≥ 16 at moderate-to-high SNR):

        BER ≈ (2/k) · (1 - 1/√M) · erfc(√(3 · Es/N0 / (2·(M-1))))

    where k = log₂(M), valid for M = 4, 16, 64, …

    Note: the argument to erfc uses Es/N0 directly (not Eb/N0 = Es/N0 / k).
    """
    k = bits_per_symbol(m)
    sqrt_m = int(np.sqrt(m))
    ber = (2.0 / k) * (1.0 - 1.0 / sqrt_m) * erfc(
        np.sqrt(3.0 * es_n0_linear / (2.0 * (m - 1)))
    )
    return float(ber)


# ---------------------------------------------------------------------------
# OFDM AWGN BER test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "mod_order, snr_db",
    [
        (4, 6.0),    # QPSK — BER ~ 2e-2
        (4, 8.0),    # QPSK — BER ~ 2e-3
        (4, 10.0),   # QPSK — BER ~ 5e-5
    ],
)
def test_ofdm_awgn_ber_matches_theory(mod_order: int, snr_db: float):
    """Measured OFDM BER over AWGN matches theoretical within 0.5 dB.

    We compare the measured BER to the theory at the *same* Es/N0. Instead
    of checking the absolute BER value, we verify that the SNR penalty
    (the extra dB of SNR needed to achieve the measured BER via theory)
    is less than 0.5 dB.
    """
    n_sub = 128
    cp_len = 16
    n_guard = 4
    n_frames = 512  # enough frames to collect ~10⁵+ bits

    ofdm = OFDMConfig(
        n_subcarriers=n_sub,
        cp_len=cp_len,
        n_guard=n_guard,
    )
    tx = OFDMTransmitter(
        modulation_order=mod_order,
        symbol_rate_hz=32e9,
        n_symbols=n_frames,
        ofdm=ofdm,
        seed=42,
    )
    rx = OFDMReceiver(modulation_order=mod_order, ofdm=ofdm)

    sent = tx.generate()

    # Add AWGN.
    snr_linear = 10.0 ** (snr_db / 10.0)
    n_data = ofdm.n_data
    # σ² = N / (n_data · SNR_sub) — see module docstring.
    noise_var = n_sub / (n_data * snr_linear)
    rng = np.random.default_rng(123)
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(sent.field.shape)
        + 1j * rng.standard_normal(sent.field.shape)
    )
    noisy_field = sent.field + noise

    rcv = rx.receive(noisy_field)

    # Measured BER.
    n_bits = sent.bits.size
    n_errors = int(np.sum(sent.bits != rcv.bits))
    measured_ber = n_errors / n_bits

    # Skip if too few errors for a reliable estimate (< 20 errors).
    if n_errors < 20:
        # At very high SNR we may get < 20 errors; just verify BER is low.
        assert measured_ber < 1e-3
        return

    # Theoretical BER at this SNR.
    theory_ber = _qam_ber_theory(mod_order, snr_linear)

    # Find the SNR penalty: what additional SNR would be needed so that the
    # theoretical BER equals our measured BER?  We invert: find SNR_theory
    # such that _qam_ber_theory(mod_order, SNR_theory) == measured_ber.
    # Then penalty_dB = 10*log10(SNR_theory) - snr_db.
    # Use a simple bisection (BER is monotonically decreasing in SNR).
    lo, hi = 0.1, 10.0 ** (snr_db / 10.0 + 3.0)  # generous bracket
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _qam_ber_theory(mod_order, mid) > measured_ber:
            lo = mid
        else:
            hi = mid
    snr_for_measured = 10.0 * np.log10((lo + hi) / 2.0)
    penalty_db = snr_for_measured - snr_db

    assert abs(penalty_db) < 0.5, (
        f"OFDM AWGN BER penalty {penalty_db:.2f} dB exceeds 0.5 dB "
        f"(M={mod_order}, SNR={snr_db} dB, "
        f"measured BER={measured_ber:.3e}, theory BER={theory_ber:.3e})"
    )
