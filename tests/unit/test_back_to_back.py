"""Back-to-back transmitter→receiver tests (no fiber, no noise)."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.receiver import SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_noiseless_b2b_recovers_all_symbols(order: int):
    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=32e9,
        n_symbols=4096,
        samples_per_symbol=8,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=0,
    )
    sent = tx.generate()

    rx = SingleCarrierReceiver(
        modulation_order=order,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
    )
    received = rx.receive(sent.field)

    # Skip pulse-shaping transients at the edges; the steady state must be
    # bit-perfect because the cascaded RRC pair is Nyquist with this config.
    skip = tx.pulse_span_symbols
    bps = int(np.log2(order))
    sent_bits = sent.bits.reshape(-1, bps)[skip:-skip]
    rx_bits = received.bits.reshape(-1, bps)[skip:-skip]
    n_errors = int(np.sum(sent_bits != rx_bits))
    assert n_errors == 0, f"BER must be 0 in noiseless b2b, got {n_errors} bit errors"


def test_dt_s_consistent_with_symbol_rate():
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=10e9, n_symbols=10, samples_per_symbol=4)
    assert tx.dt_s == pytest.approx(1.0 / (10e9 * 4))
    sig = tx.generate()
    assert sig.field.shape == (10 * 4,)
    assert sig.dt_s == tx.dt_s


def test_invalid_samples_per_symbol_rejected():
    with pytest.raises(ValueError, match="samples_per_symbol"):
        SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=10, samples_per_symbol=1)


def test_seed_makes_generation_deterministic():
    tx_a = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=1e9, n_symbols=64, seed=42)
    tx_b = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=1e9, n_symbols=64, seed=42)
    np.testing.assert_array_equal(tx_a.generate().bits, tx_b.generate().bits)
