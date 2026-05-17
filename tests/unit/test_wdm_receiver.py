"""WDMReceiver subclass — single-pol and dual-pol × WDM demux.

The end-to-end BER tests live here (rather than in ``test_wdm.py``) because
they exercise both the new WDMTransmitter and the new WDMReceiver in concert.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import bits_per_symbol
from hpcom.pilots import PilotConfig
from hpcom.receiver import Receiver, WDMReceiver
from hpcom.transmitter import WDMTransmitter
from hpcom.wdm import WDMConfig


def test_wdm_receiver_inherits_from_receiver():
    rx = WDMReceiver(
        modulation_order=4,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    assert isinstance(rx, Receiver)


def test_wdm_receiver_requires_dt_s():
    rx = WDMReceiver(
        modulation_order=4,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    with pytest.raises(ValueError, match="dt_s is required"):
        rx.receive(np.zeros(64, dtype=np.complex128))


def test_wdm_receiver_rejects_out_of_range_channel():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    with pytest.raises(ValueError, match="wdm_channel"):
        WDMReceiver(modulation_order=4, wdm=cfg, wdm_channel=3)
    with pytest.raises(ValueError, match="wdm_channel"):
        WDMReceiver(modulation_order=4, wdm=cfg, wdm_channel=-1)


def test_wdm_receiver_requires_wdm_field():
    with pytest.raises(TypeError):
        WDMReceiver(modulation_order=4)  # type: ignore[call-arg]


@pytest.mark.parametrize("ch", [0, 1, 2])
def test_dual_pol_wdm_b2b_each_pol_each_channel_ber_zero(ch: int):
    """Real end-to-end BER on dual-pol × WDM. Each (pol, channel) recovers
    its bits exactly in the noiseless steady-state region."""
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=1024,
        samples_per_symbol=16,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=0,
        polarization="dual",
        wdm=cfg,
    )
    sent = tx.generate()
    rx = WDMReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        polarization="dual",
        wdm=cfg,
        wdm_channel=ch,
    )
    received = rx.receive(sent.field, dt_s=sent.dt_s)
    assert received.symbols.shape == (2, tx.n_symbols)

    k = bits_per_symbol(tx.modulation_order)
    skip = tx.pulse_span_symbols
    for pol in (0, 1):
        sent_bits = sent.bits[pol, ch].reshape(-1, k)[skip:-skip]
        rx_bits = received.bits[pol].reshape(-1, k)[skip:-skip]
        n_err = int(np.sum(sent_bits != rx_bits))
        assert n_err == 0, (
            f"dual-pol × WDM: pol={pol} ch={ch} got {n_err} bit errors"
        )


def test_dual_pol_wdm_pilot_recovery_per_pol():
    """Pilots placed at the same positions on every (pol, channel) are
    recovered at the receiver for the selected channel, on both pols."""
    cfg_wdm = WDMConfig(n_channels=3, spacing_hz=50e9)
    cfg_p = PilotConfig(period=8, symbol=1 + 0j)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=16,
        pulse_span_symbols=32,
        seed=3,
        polarization="dual",
        wdm=cfg_wdm,
        pilots=cfg_p,
    )
    sent = tx.generate()
    for ch in range(cfg_wdm.n_channels):
        rx = WDMReceiver(
            modulation_order=4,
            samples_per_symbol=tx.samples_per_symbol,
            roll_off=tx.roll_off,
            pulse_span_symbols=tx.pulse_span_symbols,
            polarization="dual",
            wdm=cfg_wdm,
            wdm_channel=ch,
            pilots=cfg_p,
        )
        received = rx.receive(sent.field, dt_s=sent.dt_s)
        assert received.pilot_symbols is not None
        assert received.pilot_symbols.shape == (2, cfg_p.n_pilots(tx.n_symbols))
        interior = received.pilot_symbols[:, 1:-1]
        np.testing.assert_allclose(interior, cfg_p.symbol, atol=2e-2)


def test_wdm_receiver_single_pol_matches_old_behavior():
    """Single-pol WDMReceiver gives 1-D outputs (same shape as legacy
    Receiver(wdm=…) before the refactor)."""
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=16,
        pulse_span_symbols=32,
        seed=1,
        wdm=cfg,
    )
    sent = tx.generate()
    rx = WDMReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        wdm=cfg,
        wdm_channel=1,
    )
    received = rx.receive(sent.field, dt_s=sent.dt_s)
    assert received.symbols.ndim == 1
    assert received.bits.ndim == 1
    assert received.polarization == "single"
