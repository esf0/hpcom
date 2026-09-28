"""Transmitter extras (Test C): deterministic inputs, spectrum, power.

These tests use the ``Transmitter.generate(bits=…, symbols=…)`` deterministic
injection path to verify that what we feed in is exactly what comes out, both
in the time domain (after the matched filter) and the frequency domain.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import bits_per_symbol, bits_to_symbols
from hpcom.modulation.constellations import square_qam
from hpcom.receiver import SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter


@pytest.mark.parametrize("order", [4, 16, 64])
def test_fed_bits_round_trip_exactly(order: int):
    """`bits=` parameter → Tx field → Rx demap recovers the same bits in the
    steady-state region. This is the explicit version of B2B determinism."""
    k = bits_per_symbol(order)
    n = 1024
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=n * k, dtype=np.uint8)

    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=10e9,
        n_symbols=n,
        samples_per_symbol=8,
        roll_off=0.1,
        pulse_span_symbols=32,
    )
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)

    rx = SingleCarrierReceiver(
        modulation_order=order,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
    )
    received = rx.receive(sent.field)

    skip = tx.pulse_span_symbols
    sent_bits = sent.bits.reshape(-1, k)[skip:-skip]
    rx_bits = received.bits.reshape(-1, k)[skip:-skip]
    np.testing.assert_array_equal(sent_bits, rx_bits)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_fed_symbols_round_trip_exactly(order: int):
    """`symbols=` parameter (custom or non-constellation values) propagates
    through the RRC pair and is recovered at the sampling instants."""
    n = 1024
    constellation, _ = square_qam(order)
    rng = np.random.default_rng(7)
    symbols = constellation[rng.integers(0, order, size=n)]

    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=10e9,
        n_symbols=n,
        samples_per_symbol=8,
        pulse_span_symbols=32,
    )
    sent = tx.generate(symbols=symbols)
    np.testing.assert_allclose(sent.symbols, symbols, atol=1e-12)

    rx = SingleCarrierReceiver(
        modulation_order=order,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
    )
    received = rx.receive(sent.field)
    skip = tx.pulse_span_symbols
    # Finite-span RRC pair leaks ~1e-3 per neighbour into each symbol; with
    # span=32 the coherent worst case is a few %, so use atol=2e-2 for the
    # raw complex amplitude. Bit-level recovery (which uses hard decision)
    # is tested separately and is exact.
    np.testing.assert_allclose(received.symbols[skip:-skip], symbols[skip:-skip], atol=2e-2)


def test_constant_symbol_produces_cw_carrier():
    """If every symbol is the same complex value, the post-matched-filter
    samples must all equal that value (DC carrier at baseband), within the
    coherent leakage of the truncated RRC cascade."""
    n = 256
    constant = complex(0.5, -0.3)
    symbols = np.full(n, constant, dtype=np.complex128)
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=n,
        samples_per_symbol=8,
        pulse_span_symbols=64,  # longer span → less coherent leakage
    )
    sent = tx.generate(symbols=symbols)
    rx = SingleCarrierReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
    )
    received = rx.receive(sent.field)
    skip = tx.pulse_span_symbols
    # With pulse_span=64 sps=8 roll_off=0.1, coherent leakage from constant
    # neighbours stays below 2%. Tighter tolerances need a much longer filter.
    np.testing.assert_allclose(received.symbols[skip:-skip], constant, atol=2e-2)


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_average_field_power_close_to_constellation_mean(order: int):
    """For a unit-power constellation, the average sample power of the
    pulse-shaped field equals E[|s|²] / sps = 1/sps to within 5%.

    The 1/sps factor comes from upsampling by zero-stuffing — only one in
    every ``sps`` taps carries a symbol, so the time-averaged power is
    diluted accordingly. (Power *per symbol* remains unit; power *per sample*
    drops by sps.)
    """
    n = 4096
    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=10e9,
        n_symbols=n,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=0,
    )
    sent = tx.generate()
    avg_power = float(np.mean(np.abs(sent.field) ** 2))
    expected = 1.0 / tx.samples_per_symbol
    np.testing.assert_allclose(avg_power, expected, rtol=0.05)


def test_field_spectrum_is_band_limited_to_rs_times_one_plus_rolloff():
    """RRC pulse shape rolls off to ≈0 beyond Rs·(1+β)/2 around DC. Verify
    that >99% of the field energy lives within that band."""
    rs = 10e9
    rolloff = 0.1
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=rs,
        n_symbols=8192,
        samples_per_symbol=8,
        roll_off=rolloff,
        pulse_span_symbols=32,
        seed=0,
    )
    sent = tx.generate()
    fs = 1.0 / sent.dt_s
    spectrum = np.fft.fft(sent.field)
    psd = np.abs(spectrum) ** 2
    freqs = np.fft.fftfreq(sent.field.size, d=sent.dt_s)
    cutoff = rs * (1.0 + rolloff) / 2.0
    in_band = psd[np.abs(freqs) <= cutoff].sum()
    total = psd.sum()
    fraction = in_band / total
    assert fraction > 0.99, (
        f"only {fraction:.3%} of energy is in [-Rs(1+β)/2, +Rs(1+β)/2] "
        f"({cutoff:.2e} Hz of fs={fs:.2e} Hz)"
    )


def test_generate_rejects_both_bits_and_symbols():
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8)
    with pytest.raises(ValueError, match="at most one"):
        tx.generate(bits=np.zeros(16, dtype=np.uint8), symbols=np.ones(8, dtype=np.complex128))


def test_generate_rejects_wrong_shape_bits():
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=1e9, n_symbols=10)
    with pytest.raises(ValueError, match=r"`bits` must have shape"):
        tx.generate(bits=np.zeros(7, dtype=np.uint8))  # not a multiple


def test_generate_rejects_wrong_shape_symbols():
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=10)
    with pytest.raises(ValueError, match=r"`symbols` must have shape"):
        tx.generate(symbols=np.zeros(11, dtype=np.complex128))


def test_symbols_input_demaps_off_grid_to_nearest_bits():
    """When non-constellation symbols are fed, the returned ``bits`` come from
    a hard-decision demap. Slightly noisy versions of constellation points
    should demap to the same bits as the clean version."""
    order = 16
    constellation, labels = square_qam(order)
    # Pick the first 8 constellation points, perturb each by a tiny noise.
    clean = constellation[:8]
    noisy = clean + 1e-3 * (
        np.random.default_rng(0).standard_normal(8)
        + 1j * np.random.default_rng(1).standard_normal(8)
    )
    tx = SingleCarrierTransmitter(modulation_order=order, symbol_rate_hz=1e9, n_symbols=8)
    sent_clean = tx.generate(symbols=clean)
    sent_noisy = tx.generate(symbols=noisy)
    # Demapped bits must match (noise is below the decision region radius).
    np.testing.assert_array_equal(sent_clean.bits, sent_noisy.bits)
    # And those bits must match the labels of the original 8 points.
    k = bits_per_symbol(order)
    expected_bits = np.concatenate([labels[i] for i in range(8)])
    np.testing.assert_array_equal(sent_clean.bits, expected_bits)
    # Symbols field, in contrast, keeps the *fed* value (off-grid is preserved).
    np.testing.assert_allclose(sent_clean.symbols, clean, atol=1e-12)
    np.testing.assert_allclose(sent_noisy.symbols, noisy, atol=1e-12)
    # Sanity: confirm bits_to_symbols(sent_clean.bits) recovers the clean
    # constellation points (round-trip check).
    np.testing.assert_allclose(
        bits_to_symbols(sent_clean.bits.reshape(-1)[: 8 * k], order),
        clean,
        atol=1e-12,
    )
