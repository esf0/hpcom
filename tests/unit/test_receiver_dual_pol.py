"""Dual-polarisation Receiver demodulation.

Companion to ``test_transmitter_dual_pol.py``. The dual-pol Receiver consumes a
``(2, n_samples)`` field and runs the matched-filter / demap pipeline once per
polarisation, returning 2-D bits / symbols / decisions.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import bits_per_symbol
from hpcom.pilots import PilotConfig
from hpcom.receiver import Receiver, SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter, Transmitter
from hpcom.wdm import WDMConfig


def _b2b_pair(*, order: int = 4, n: int = 512) -> tuple[Transmitter, Receiver]:
    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=10e9,
        n_symbols=n,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=0,
        polarization="dual",
    )
    rx = SingleCarrierReceiver(
        modulation_order=order,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        polarization="dual",
    )
    return tx, rx


def test_dual_pol_receiver_returns_2d_shapes():
    tx, rx = _b2b_pair()
    sent = tx.generate()
    received = rx.receive(sent.field)
    k = bits_per_symbol(tx.modulation_order)
    assert received.symbols.shape == (2, tx.n_symbols)
    assert received.bits.shape == (2, tx.n_symbols * k)
    assert received.decisions.shape == (2, tx.n_symbols)
    assert received.polarization == "dual"


@pytest.mark.parametrize("order", [4, 16, 64])
def test_dual_pol_b2b_each_pol_ber_zero(order: int):
    tx, rx = _b2b_pair(order=order, n=1024)
    sent = tx.generate()
    received = rx.receive(sent.field)
    k = bits_per_symbol(order)
    skip = tx.pulse_span_symbols
    for pol in (0, 1):
        sent_bits = sent.bits[pol].reshape(-1, k)[skip:-skip]
        rx_bits = received.bits[pol].reshape(-1, k)[skip:-skip]
        assert int(np.sum(sent_bits != rx_bits)) == 0


def test_dual_pol_streams_are_independent_at_receiver():
    """X bits and Y bits are different at the receiver (because Tx made them
    different); the Rx must not accidentally couple them."""
    tx, rx = _b2b_pair(n=512)
    sent = tx.generate()
    received = rx.receive(sent.field)
    assert not np.array_equal(received.bits[0], received.bits[1])
    assert not np.allclose(received.symbols[0], received.symbols[1])


def test_dual_pol_receiver_rejects_1d_input():
    rx = SingleCarrierReceiver(modulation_order=4, polarization="dual")
    with pytest.raises(ValueError, match="dual-pol receiver expects 2-D field"):
        rx.receive(np.zeros(64, dtype=np.complex128))


def test_single_pol_receiver_rejects_2d_input():
    rx = SingleCarrierReceiver(modulation_order=4, polarization="single")
    with pytest.raises(ValueError, match="single-pol receiver expects 1-D field"):
        rx.receive(np.zeros((2, 64), dtype=np.complex128))


def test_dual_pol_pilot_symbols_per_pol():
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=2,
        polarization="dual",
        pilots=cfg,
    )
    rx = SingleCarrierReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        polarization="dual",
        pilots=cfg,
    )
    sent = tx.generate()
    received = rx.receive(sent.field)
    assert received.pilot_symbols is not None
    assert received.pilot_symbols.shape == (2, cfg.n_pilots(tx.n_symbols))
    # Each polarisation recovers the pilot value in the steady-state region.
    interior = received.pilot_symbols[:, 1:-1]
    np.testing.assert_allclose(interior, cfg.symbol, atol=2e-2)


def test_dual_pol_data_bits_strips_pilots_per_pol():
    cfg = PilotConfig(period=4, symbol=1 + 0j)
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=8,
        pulse_span_symbols=16,
        seed=0,
        polarization="dual",
        pilots=cfg,
    )
    rx = SingleCarrierReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        polarization="dual",
        pilots=cfg,
    )
    sent = tx.generate()
    received = rx.receive(sent.field)
    n_data = cfg.n_data(tx.n_symbols)
    k = 2
    assert sent.data_bits().shape == (2, n_data * k)
    assert received.data_bits().shape == (2, n_data * k)


def test_dual_pol_plus_wdm_uses_wdm_receiver():
    """Dual-pol × WDM demux is supported via WDMReceiver."""
    from hpcom.wdm import WDMReceiver

    rx = WDMReceiver(
        modulation_order=4,
        polarization="dual",
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        wdm_channel=1,
    )
    assert rx.polarization == "dual"
    assert rx.wdm_channel == 1


def test_dual_pol_invalid_polarization_rejected():
    with pytest.raises(ValueError, match="polarization"):
        SingleCarrierReceiver(modulation_order=4, polarization="triple")  # type: ignore[arg-type]


def test_dual_pol_received_signal_has_polarization_field():
    tx, rx = _b2b_pair()
    received = rx.receive(tx.generate().field)
    assert received.polarization == "dual"

    tx_s = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8, seed=0)
    rx_s = SingleCarrierReceiver(modulation_order=4, samples_per_symbol=tx_s.samples_per_symbol,
                    roll_off=tx_s.roll_off, pulse_span_symbols=tx_s.pulse_span_symbols)
    received_s = rx_s.receive(tx_s.generate().field)
    assert received_s.polarization == "single"
