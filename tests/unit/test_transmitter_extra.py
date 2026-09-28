"""Transmitter: deterministic inputs, calibration, spectrum, launch power."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from helpers import PULSE_MODES, sc_pair, trim
from hpcom import SingleCarrierTransmitter
from hpcom.modulation import bits_to_symbols
from hpcom.modulation.constellations import square_qam
from hpcom.units import dbm_to_w

# Worst-case ISI of the truncated span-32 RRC pair (β = 0.1, see _pulse.py);
# the exact RRC is ISI-free to rounding.
_ISI_ATOL = {"exact-circular": 1e-12, "fir-circular": 2e-2, "fir-linear": 2e-2}


def _mode_id(pulse: dict) -> str:
    if not pulse:
        return "exact-circular"
    return "fir-linear" if pulse.get("convolution") == "linear" else "fir-circular"


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("order", [4, 16, 64])
def test_fed_bits_are_transmitted_and_recovered(order, pulse):
    tx, rx = sc_pair(modulation_order=order, samples_per_symbol=8, **pulse)
    bits = np.random.default_rng(42).integers(0, 2, tx.n_bits_per_stream, dtype=np.uint8)
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)
    np.testing.assert_allclose(sent.symbols, bits_to_symbols(bits, order), rtol=0, atol=1e-15)
    received = rx.receive(sent.field, sent.dt_s)
    k, skip = tx.bits_per_symbol, tx.transient_symbols
    np.testing.assert_array_equal(trim(received.bits, k, skip), trim(bits, k, skip))


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("order", [4, 16, 64])
def test_fed_symbols_reach_the_receiver_with_absolute_calibration(order, pulse):
    """Matched-filter output (no equaliser) = √P_pol · fed symbol, up to pulse ISI."""
    tx, rx = sc_pair(
        modulation_order=order, samples_per_symbol=8, rx_kwargs={"equalize": "none"}, **pulse
    )
    points, _ = square_qam(order)
    symbols = points[np.random.default_rng(7).integers(0, order, tx.n_symbols)]
    sent = tx.generate(symbols=symbols)
    np.testing.assert_array_equal(sent.symbols, symbols)
    out = rx.receive(sent.field, sent.dt_s).symbols / np.sqrt(tx.launch_power_per_polarization_w)
    skip = tx.transient_symbols
    np.testing.assert_allclose(
        trim(out, 1, skip), trim(symbols, 1, skip), rtol=0, atol=_ISI_ATOL[_mode_id(pulse)]
    )


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_constant_symbols_give_a_cw_carrier(pulse):
    constant = complex(0.5, -0.3)
    tx, rx = sc_pair(
        modulation_order=4,
        n_symbols=256,
        samples_per_symbol=8,
        rx_kwargs={"equalize": "none"},
        **pulse,
    )
    sent = tx.generate(symbols=np.full(tx.n_symbols, constant))
    out = rx.receive(sent.field, sent.dt_s).symbols / np.sqrt(tx.launch_power_per_polarization_w)
    skip = tx.transient_symbols
    np.testing.assert_allclose(
        trim(out, 1, skip), constant, rtol=0, atol=_ISI_ATOL[_mode_id(pulse)]
    )


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("order", [4, 16, 64, 256])
@pytest.mark.parametrize("power_dbm", [-10.0, 0.0, 3.0])
def test_field_power_is_the_launch_power(order, power_dbm, pulse):
    """Expected power = launch power; one realisation is within O(1/√n)."""
    tx, _ = sc_pair(
        modulation_order=order,
        n_symbols=8192,
        seed=0,
        launch_power_per_channel_dbm=power_dbm,
        **pulse,
    )
    measured = tx.generate().mean_power_w
    np.testing.assert_allclose(measured, dbm_to_w(power_dbm), rtol=0.03)


def test_default_launch_power_is_zero_dbm_per_channel():
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=8)
    assert tx.launch_power_per_channel_w == pytest.approx(1e-3)
    assert tx.launch_power_total_w == pytest.approx(1e-3)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_launch_power_is_split_between_polarisations(polarization):
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=32e9,
        n_symbols=8192,
        seed=0,
        polarization=polarization,
        launch_power_per_channel_dbm=3.0,
    )
    n_pol = 2 if polarization == "dual" else 1
    assert tx.launch_power_per_polarization_w == pytest.approx(dbm_to_w(3.0) / n_pol)
    sig = tx.generate()
    per_pol = np.mean(np.abs(sig.field.reshape(n_pol, -1)) ** 2, axis=-1)
    np.testing.assert_allclose(per_pol, dbm_to_w(3.0) / n_pol, rtol=0.03)
    np.testing.assert_allclose(sig.mean_power_w, dbm_to_w(3.0), rtol=0.03)


def test_total_power_equals_per_channel_power_for_a_single_channel():
    a = SingleCarrierTransmitter(
        modulation_order=4, symbol_rate_hz=32e9, n_symbols=8, launch_power_total_dbm=2.0
    )
    assert a.launch_power_per_channel_w == pytest.approx(dbm_to_w(2.0))


def test_both_power_fields_is_an_error():
    with pytest.raises(ValidationError, match="at most one"):
        SingleCarrierTransmitter(
            modulation_order=4,
            symbol_rate_hz=32e9,
            n_symbols=8,
            launch_power_per_channel_dbm=0.0,
            launch_power_total_dbm=0.0,
        )


def test_with_launch_power_switches_between_the_two_inputs():
    tx = SingleCarrierTransmitter(
        modulation_order=4, symbol_rate_hz=32e9, n_symbols=8, launch_power_per_channel_dbm=1.0
    )
    total = tx.with_launch_power(total_dbm=5.0)
    assert total.launch_power_per_channel_dbm is None
    assert total.launch_power_per_channel_w == pytest.approx(dbm_to_w(5.0))
    back = total.with_launch_power(per_channel_dbm=-2.0)
    assert back.launch_power_total_dbm is None
    with pytest.raises(TypeError, match="exactly one"):
        tx.with_launch_power()


def test_exact_rrc_spectrum_is_strictly_band_limited():
    """No energy at all beyond R_s(1+β)/2 (to rounding) for the exact RRC."""
    tx, _ = sc_pair(modulation_order=16, n_symbols=4096, samples_per_symbol=8, seed=0)
    sig = tx.generate()
    psd = np.abs(np.fft.fft(sig.field)) ** 2
    f = np.fft.fftfreq(sig.field.size, sig.dt_s)
    outside = psd[np.abs(f) > tx.symbol_rate_hz * (1 + tx.roll_off) / 2 * (1 + 1e-9)]
    assert outside.max() < 1e-26 * psd.max()


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_spectrum_holds_99_percent_of_energy_in_band(pulse):
    tx, _ = sc_pair(modulation_order=16, n_symbols=8192, samples_per_symbol=8, seed=0, **pulse)
    sig = tx.generate()
    psd = np.abs(np.fft.fft(sig.field)) ** 2
    f = np.fft.fftfreq(sig.field.size, sig.dt_s)
    cutoff = tx.symbol_rate_hz * (1 + tx.roll_off) / 2
    assert psd[np.abs(f) <= cutoff].sum() / psd.sum() > 0.99


def test_generate_rejects_both_bits_and_symbols():
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8)
    with pytest.raises(ValueError, match="at most one"):
        tx.generate(bits=np.zeros(16, dtype=np.uint8), symbols=np.ones(8))


@pytest.mark.parametrize("n_bits", [7, 39, 41])
def test_generate_rejects_wrong_number_of_bits(n_bits):
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=1e9, n_symbols=10)
    with pytest.raises(ValueError, match=r"`bits` must have shape \(40,\)"):
        tx.generate(bits=np.zeros(n_bits, dtype=np.uint8))


def test_generate_rejects_non_binary_bits():
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=4)
    with pytest.raises(ValueError, match="only 0 and 1"):
        tx.generate(bits=np.array([0, 1, 2, 0, 1, 1, 0, 0]))


def test_generate_rejects_wrong_number_of_symbols():
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=10)
    with pytest.raises(ValueError, match=r"`symbols` must have shape \(10,\)"):
        tx.generate(symbols=np.zeros(11, dtype=np.complex128))


def test_fed_symbols_are_kept_off_grid_and_bits_come_from_the_nearest_point():
    order = 16
    points, labels = square_qam(order)
    clean = points[:8]
    noisy = clean + 1e-3 * (np.random.default_rng(0).standard_normal(8) + 1j)
    tx = SingleCarrierTransmitter(modulation_order=order, symbol_rate_hz=1e9, n_symbols=8)
    sent_clean, sent_noisy = tx.generate(symbols=clean), tx.generate(symbols=noisy)
    np.testing.assert_array_equal(sent_noisy.bits, sent_clean.bits)
    np.testing.assert_array_equal(sent_clean.bits, labels[:8].ravel())
    np.testing.assert_array_equal(sent_noisy.symbols, noisy)


def test_signal_arrays_are_read_only_and_caller_arrays_stay_writeable():
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8)
    bits = np.zeros(16, dtype=np.uint8)
    sig = tx.generate(bits=bits)
    for arr in (sig.field, sig.bits, sig.symbols):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1
    bits[0] = 1  # the caller's array is untouched and writeable
    assert sig.bits[0] == 0


def test_linear_convolution_requires_a_finite_span():
    with pytest.raises(ValidationError, match="finite"):
        SingleCarrierTransmitter(
            modulation_order=4, symbol_rate_hz=1e9, n_symbols=64, convolution="linear"
        )


def test_span_must_be_shorter_than_the_block():
    with pytest.raises(ValidationError, match="smaller than"):
        SingleCarrierTransmitter(
            modulation_order=4, symbol_rate_hz=1e9, n_symbols=32, pulse_span_symbols=32
        )


def test_transient_symbols_only_for_linear_convolution():
    kw = dict(modulation_order=4, symbol_rate_hz=1e9, n_symbols=128)
    assert SingleCarrierTransmitter(**kw).transient_symbols == 0
    assert SingleCarrierTransmitter(**kw, pulse_span_symbols=16).transient_symbols == 0
    assert (
        SingleCarrierTransmitter(
            **kw, pulse_span_symbols=16, convolution="linear"
        ).transient_symbols
        == 16
    )


def test_unknown_keyword_is_rejected():
    with pytest.raises(ValidationError, match="modulaton_order"):
        SingleCarrierTransmitter(modulaton_order=16, symbol_rate_hz=1e9, n_symbols=8)  # typo
