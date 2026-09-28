"""Pilot symbols (Test G) — insertion, extraction, BER conventions.

Verifies that:

- ``PilotConfig`` validates its period
- Tx places the configured pilot symbol at the right positions
- Rx returns the post-matched-filter samples at pilot positions
- Both BER conventions are supported (α: data_bits stripped; β: full bits)
- Pilots survive WDM mux/demux per channel
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import bits_per_symbol
from hpcom.pilots import PilotConfig
from hpcom.receiver import SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter
from hpcom.wdm import WDMConfig, WDMReceiver, WDMTransmitter


def test_pilot_config_rejects_period_below_two():
    with pytest.raises(ValueError, match="period"):
        PilotConfig(period=1)


def test_pilot_indices_and_counts():
    cfg = PilotConfig(period=4)
    n = 16
    np.testing.assert_array_equal(cfg.pilot_indices(n), [0, 4, 8, 12])
    np.testing.assert_array_equal(cfg.data_indices(n), [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
    assert cfg.n_pilots(n) == 4
    assert cfg.n_data(n) == 12

    # n_symbols not a clean multiple of period.
    n2 = 14
    np.testing.assert_array_equal(cfg.pilot_indices(n2), [0, 4, 8, 12])
    assert cfg.n_pilots(n2) + cfg.n_data(n2) == n2


def test_tx_inserts_configured_pilot_at_expected_positions():
    pilot_val = complex(0.7, -0.4)
    cfg = PilotConfig(period=8, symbol=pilot_val)
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=8,
        pulse_span_symbols=16,
        seed=0,
        pilots=cfg,
    )
    sent = tx.generate()
    pilot_positions = cfg.pilot_indices(tx.n_symbols)
    np.testing.assert_allclose(sent.symbols[pilot_positions], pilot_val, atol=1e-12)
    data_positions = cfg.data_indices(tx.n_symbols)
    assert not np.any(np.isclose(sent.symbols[data_positions], pilot_val))
    k = bits_per_symbol(tx.modulation_order)
    bits_2d = sent.bits.reshape(-1, k)
    assert np.all(bits_2d[pilot_positions] == 0)


def test_rx_extracts_pilot_amplitudes_in_b2b():
    """Noiseless B2B: received pilot_symbols equal the configured pilot value
    (within finite-RRC leakage tolerance)."""
    pilot_val = complex(1.0, 0.0)
    cfg = PilotConfig(period=10, symbol=pilot_val)
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=500,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=1,
        pilots=cfg,
    )
    sent = tx.generate()

    rx = SingleCarrierReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        pilots=cfg,
    )
    received = rx.receive(sent.field)
    assert received.pilot_symbols is not None
    assert received.pilot_symbols.shape == (cfg.n_pilots(tx.n_symbols),)
    # Skip the first / last pilots in case they fall inside the edge transient.
    interior = received.pilot_symbols[1:-1]
    np.testing.assert_allclose(interior, pilot_val, atol=2e-2)


def test_option_alpha_data_bits_round_trip():
    """Option α: TransmittedSignal.data_bits() round-trips through Rx.data_bits()."""
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=2,
        pilots=cfg,
    )
    sent = tx.generate()
    rx = SingleCarrierReceiver(
        modulation_order=16,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        pilots=cfg,
    )
    received = rx.receive(sent.field)

    # data_bits has shape (n_data * k,) — pilots stripped.
    k = bits_per_symbol(tx.modulation_order)
    n_data = cfg.n_data(tx.n_symbols)
    assert sent.data_bits().shape == (n_data * k,)
    assert received.data_bits().shape == (n_data * k,)

    # Mask out the pulse-edge symbols so finite-span RRC ISI doesn't leak in.
    # We compare the data bits in the steady-state interior.
    skip = tx.pulse_span_symbols
    # Map symbol-index skip → bit positions in the *data* stream by counting
    # how many data slots lie in [skip, n - skip).
    data_idx = cfg.data_indices(tx.n_symbols)
    interior_mask = (data_idx >= skip) & (data_idx < tx.n_symbols - skip)
    sent_view = sent.data_bits().reshape(-1, k)[interior_mask]
    rx_view = received.data_bits().reshape(-1, k)[interior_mask]
    n_err = int(np.sum(sent_view != rx_view))
    assert n_err == 0, f"data-bit BER must be 0 in noiseless B2B, got {n_err} errors"


def test_option_beta_full_bits_carry_pilot_zeros():
    """Option β: full bits array round-trips structurally (pilot slots are
    zero on Tx; receiver hard-decisions them but caller can mask externally)."""
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx = SingleCarrierTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=256,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=3,
        pilots=cfg,
    )
    sent = tx.generate()
    rx = SingleCarrierReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
        pilots=cfg,
    )
    received = rx.receive(sent.field)

    # Same shape as the no-pilot case.
    k = bits_per_symbol(tx.modulation_order)
    assert sent.bits.shape == (tx.n_symbols * k,)
    assert received.bits.shape == (tx.n_symbols * k,)

    # Manually exclude pilot positions for BER; interior of the pulse window.
    skip = tx.pulse_span_symbols
    pilot_idx = cfg.pilot_indices(tx.n_symbols)
    interior_data_idx = np.array(
        [i for i in range(skip, tx.n_symbols - skip) if i not in set(pilot_idx)],
        dtype=np.intp,
    )
    sent_view = sent.bits.reshape(-1, k)[interior_data_idx]
    rx_view = received.bits.reshape(-1, k)[interior_data_idx]
    assert int(np.sum(sent_view != rx_view)) == 0


def test_pilots_in_wdm_per_channel_recovery():
    """3-channel WDM + pilots: each channel's pilot_symbols recover the
    configured value at the receiver after frequency-shift and demux."""
    cfg_wdm = WDMConfig(n_channels=3, spacing_hz=50e9)
    cfg_pilot = PilotConfig(period=8, symbol=1 + 0j)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=16,
        pulse_span_symbols=32,
        seed=4,
        wdm=cfg_wdm,
        pilots=cfg_pilot,
    )
    sent = tx.generate()
    # Per-channel pilot insertion check.
    pilot_idx = cfg_pilot.pilot_indices(tx.n_symbols)
    for ch in range(cfg_wdm.n_channels):
        np.testing.assert_allclose(sent.symbols[ch, pilot_idx], cfg_pilot.symbol, atol=1e-12)

    for ch in range(cfg_wdm.n_channels):
        rx = WDMReceiver(
            modulation_order=4,
            samples_per_symbol=tx.samples_per_symbol,
            roll_off=tx.roll_off,
            pulse_span_symbols=tx.pulse_span_symbols,
            wdm=cfg_wdm,
            wdm_channel=ch,
            pilots=cfg_pilot,
        )
        received = rx.receive(sent.field, dt_s=sent.dt_s)
        assert received.pilot_symbols is not None
        interior = received.pilot_symbols[1:-1]
        np.testing.assert_allclose(interior, cfg_pilot.symbol, atol=2e-2)


def test_no_pilots_data_bits_returns_full_bits():
    """When pilots is None, data_bits() should be the same array as bits."""
    tx = SingleCarrierTransmitter(modulation_order=4, symbol_rate_hz=1e9, n_symbols=8, seed=0)
    sent = tx.generate()
    assert sent.data_bits() is sent.bits
    rx = SingleCarrierReceiver(
        modulation_order=4,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
    )
    received = rx.receive(sent.field)
    assert received.data_bits() is received.bits


def test_pilot_count_and_data_count_partition_n_symbols():
    """Internal consistency on PilotConfig partitioning."""
    cfg = PilotConfig(period=7)
    for n in [10, 49, 50, 100, 1000]:
        assert cfg.n_pilots(n) + cfg.n_data(n) == n
        assert len(cfg.pilot_indices(n)) == cfg.n_pilots(n)
        assert len(cfg.data_indices(n)) == cfg.n_data(n)
        # Indices are disjoint and cover the full range.
        all_idx = np.sort(np.concatenate([cfg.pilot_indices(n), cfg.data_indices(n)]))
        np.testing.assert_array_equal(all_idx, np.arange(n))
