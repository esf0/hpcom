"""Dual-polarisation single-carrier transmitter."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from helpers import PULSE_MODES, sc_pair, trim
from hpcom import PilotConfig, SingleCarrierTransmitter, WDMConfig, WDMTransmitter


def _tx(**kw) -> SingleCarrierTransmitter:
    base = dict(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=256, polarization="dual", seed=0
    )
    return SingleCarrierTransmitter(**{**base, **kw})


def test_shapes_and_independent_polarisations():
    tx = _tx()
    sent = tx.generate()
    assert sent.field.shape == (2, tx.n_samples)
    assert sent.symbols.shape == (2, tx.n_symbols)
    assert sent.bits.shape == (2, tx.n_bits_per_stream)
    assert tx.stream_shape == (2,)
    assert not np.array_equal(sent.bits[0], sent.bits[1])


def test_seed_makes_both_polarisations_deterministic():
    a, b = _tx(seed=5).generate(), _tx(seed=5).generate()
    np.testing.assert_array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.field, b.field)


def test_x_polarisation_equals_the_single_polarisation_stream_with_the_same_seed():
    """Stream i draws from SeedSequence(seed).spawn(...)[i]: X of dual == single."""
    single = _tx(polarization="single", seed=3).generate()
    dual = _tx(seed=3).generate()
    np.testing.assert_array_equal(dual.bits[0], single.bits)


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("order", [4, 16, 64])
def test_b2b_each_polarisation_recovers_its_own_bits(order, pulse):
    tx, rx = sc_pair(
        modulation_order=order, polarization="dual", seed=1, samples_per_symbol=8, **pulse
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    k, skip = tx.bits_per_symbol, tx.transient_symbols
    for pol in (0, 1):
        np.testing.assert_array_equal(
            trim(received.bits[pol], k, skip), trim(sent.bits[pol], k, skip)
        )


def test_deterministic_bits_input():
    n = 64
    bits = np.array([np.tile([0, 1], n), np.tile([1, 0], n)], dtype=np.uint8)
    sent = _tx(modulation_order=4, n_symbols=n).generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)
    assert not np.allclose(sent.symbols[0], sent.symbols[1])


def test_deterministic_symbols_input():
    symbols = np.zeros((2, 32), dtype=complex)
    symbols[0], symbols[1] = 1 + 0j, 1j
    sent = _tx(modulation_order=4, n_symbols=32).generate(symbols=symbols)
    np.testing.assert_array_equal(sent.symbols, symbols)


def test_pilots_are_inserted_in_both_polarisations():
    cfg = PilotConfig(period=8, symbol=0.7 + 0.3j)
    tx = _tx(modulation_order=4, n_symbols=64, pilots=cfg)
    sent = tx.generate()
    for pol in (0, 1):
        np.testing.assert_array_equal(sent.symbols[pol, cfg.pilot_indices(64)], cfg.symbol)
    assert sent.bits.shape == (2, cfg.n_data(64) * 2)


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [("bits", np.zeros(20, dtype=np.uint8)), ("symbols", np.zeros(10, dtype=complex))],
)
def test_single_polarisation_shaped_input_is_rejected(kwarg, value):
    tx = _tx(modulation_order=4, n_symbols=10)
    with pytest.raises(ValueError, match=rf"`{kwarg}` must have shape \(2, "):
        tx.generate(**{kwarg: value})


def test_dual_polarisation_wdm_shapes():
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        polarization="dual",
        seed=0,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    sent = tx.generate()
    assert sent.field.shape == (2, tx.n_samples)
    assert sent.bits.shape == (2, 3, 128)
    assert sent.symbols.shape == (2, 3, 64)
    assert sent.polarization == "dual"
    assert sent.transmitter.wdm == tx.wdm


def test_power_is_split_evenly_between_polarisations():
    sent = _tx(n_symbols=8192, launch_power_per_channel_dbm=0.0).generate()
    px, py = np.mean(np.abs(sent.field) ** 2, axis=-1)
    np.testing.assert_allclose([px, py], 0.5e-3, rtol=0.03)


def test_invalid_polarization_is_rejected():
    with pytest.raises(ValidationError, match="polarization"):
        _tx(polarization="triple")
