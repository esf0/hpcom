"""WDM (T5.4) — transmitter superposition and receiver demux."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.wdm import WDMConfig, WDMReceiver, WDMTransmitter


def test_wdm_config_rejects_lt_two_channels():
    with pytest.raises(ValueError, match="n_channels"):
        WDMConfig(n_channels=1, spacing_hz=50e9)


def test_wdm_config_rejects_non_positive_spacing():
    with pytest.raises(ValueError, match="spacing_hz"):
        WDMConfig(n_channels=3, spacing_hz=0.0)


def test_wdm_channel_frequencies_are_centered():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    np.testing.assert_allclose(
        cfg.channel_frequencies_hz(), [-50e9, 0.0, 50e9]
    )
    # Even N → symmetric around 0, no channel exactly at 0.
    cfg2 = WDMConfig(n_channels=4, spacing_hz=25e9)
    np.testing.assert_allclose(
        cfg2.channel_frequencies_hz(), [-37.5e9, -12.5e9, 12.5e9, 37.5e9]
    )


def test_transmitter_rejects_undersampled_wdm():
    # 5 channels × 50 GHz spacing → outer center at ±100 GHz; with Rs=10 GHz
    # and sps=16 the wideband fs is 160 GHz, far below the 2 · 105 GHz needed.
    with pytest.raises(ValueError, match="Nyquist"):
        WDMTransmitter(
            modulation_order=4,
            symbol_rate_hz=10e9,
            n_symbols=64,
            samples_per_symbol=16,
            wdm=WDMConfig(n_channels=5, spacing_hz=50e9),
        )


def test_three_channel_wdm_b2b_center_channel_ber_zero():
    """3 channels at 50 GHz spacing; demux center, expect BER == 0.

    Acceptance criterion for T5.4 in docs/spec/hpcom-v1-tasks.md.
    """
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    # sps must cover 2 · (50e9 + Rs·(1+ro)/2) = 2 · (50e9 + 5.5e9) = 111 GHz.
    # With Rs=10 GHz, sps=16 → fs=160 GHz, comfortable.
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=2048,
        samples_per_symbol=16,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=0,
        wdm=cfg,
    )
    sent = tx.generate()

    # Per-channel arrays.
    assert sent.bits.shape == (3, 2048 * 4)
    assert sent.symbols.shape == (3, 2048)
    assert sent.field.ndim == 1
    assert sent.wdm is cfg

    rx = WDMReceiver(
        modulation_order=16,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        wdm=cfg,
        wdm_channel=1,  # center
    )
    received = rx.receive(sent.field, dt_s=sent.dt_s)

    skip = tx.pulse_span_symbols
    bps = 4
    sent_bits = sent.bits[1].reshape(-1, bps)[skip:-skip]
    rx_bits = received.bits.reshape(-1, bps)[skip:-skip]
    n_errors = int(np.sum(sent_bits != rx_bits))
    assert n_errors == 0, f"WDM center-channel BER must be 0, got {n_errors} bit errors"


@pytest.mark.parametrize("ch", [0, 1, 2])
def test_three_channel_wdm_b2b_all_channels_ber_zero(ch: int):
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=1024,
        samples_per_symbol=16,
        roll_off=0.1,
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
        wdm_channel=ch,
    )
    received = rx.receive(sent.field, dt_s=sent.dt_s)

    skip = tx.pulse_span_symbols
    sent_bits = sent.bits[ch].reshape(-1, 2)[skip:-skip]
    rx_bits = received.bits.reshape(-1, 2)[skip:-skip]
    n_errors = int(np.sum(sent_bits != rx_bits))
    assert n_errors == 0, f"WDM channel {ch} BER must be 0, got {n_errors} bit errors"


def test_wdm_seed_makes_generation_deterministic():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    kwargs = dict(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        seed=7,
        wdm=cfg,
    )
    a = WDMTransmitter(**kwargs).generate()
    b = WDMTransmitter(**kwargs).generate()
    np.testing.assert_array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.field, b.field)


def test_receiver_rejects_out_of_range_wdm_channel():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    with pytest.raises(ValueError, match="wdm_channel"):
        WDMReceiver(modulation_order=4, wdm=cfg, wdm_channel=3)
