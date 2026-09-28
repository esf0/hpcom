"""Noiseless back-to-back transmitter → receiver (no fiber, no noise)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from helpers import PULSE_MODES, sc_pair, trim
from hpcom import SingleCarrierTransmitter


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_noiseless_b2b_recovers_every_bit(order, pulse):
    tx, rx = sc_pair(modulation_order=order, n_symbols=4096, samples_per_symbol=8, seed=0, **pulse)
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    k = tx.bits_per_symbol
    skip = tx.transient_symbols
    np.testing.assert_array_equal(trim(received.bits, k, skip), trim(sent.bits, k, skip))


@pytest.mark.parametrize("order", [4, 64])
def test_exact_rrc_b2b_is_isi_free_to_rounding(order):
    """Exact (frequency-domain) RRC: matched-filter output = √P·symbol to ~1e-14."""
    tx, rx = sc_pair(modulation_order=order, n_symbols=2048, seed=1, rx_kwargs={"equalize": "none"})
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    scaled = received.symbols / np.sqrt(tx.launch_power_per_polarization_w)
    np.testing.assert_allclose(scaled, sent.symbols, rtol=0, atol=1e-13)


def test_dt_s_and_sizes_follow_the_symbol_rate():
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=10, samples_per_symbol=4
    )
    assert tx.dt_s == pytest.approx(1.0 / 40e9)
    sig = tx.generate()
    assert sig.field.shape == (40,)
    assert sig.bits.shape == (40,)
    assert sig.symbols.shape == (10,)
    assert sig.dt_s == tx.dt_s
    assert tx.n_samples == 40


@pytest.mark.parametrize("sps", [0, 1])
def test_samples_per_symbol_below_two_is_rejected(sps):
    with pytest.raises(ValidationError, match="samples_per_symbol"):
        SingleCarrierTransmitter(
            modulation_order=4, symbol_rate_hz=1e9, n_symbols=10, samples_per_symbol=sps
        )


def test_seed_makes_generation_deterministic():
    kw = dict(modulation_order=16, symbol_rate_hz=1e9, n_symbols=64, seed=42)
    a, b = SingleCarrierTransmitter(**kw).generate(), SingleCarrierTransmitter(**kw).generate()
    np.testing.assert_array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.field, b.field)


def test_rng_argument_overrides_seed_and_is_reproducible():
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=1e9, n_symbols=64, seed=42)
    a = tx.generate(rng=np.random.default_rng(7))
    b = tx.generate(rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.bits, b.bits)
    assert not np.array_equal(a.bits, tx.generate().bits)


def test_unseeded_transmitter_draws_fresh_data():
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=1e9, n_symbols=256)
    assert not np.array_equal(tx.generate().bits, tx.generate().bits)
