"""WDMTransmitter subclass — shape conventions and dual-pol × WDM behaviour.

Complements ``test_wdm.py`` (single-pol BER) and ``test_wdm_spectrum.py``
(spectral checks) with construction-side tests for the new class, especially
the dual-pol path that this refactor unlocks.

End-to-end BER tests for dual-pol × WDM live with the future ``WDMReceiver``
commit; here we verify the Tx side with manual matched-filter recovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom._pulse import root_raised_cosine
from hpcom.modulation import bits_per_symbol, symbols_to_bits
from hpcom.pilots import PilotConfig
from hpcom.transmitter import Transmitter
from hpcom.wdm import WDMConfig, WDMTransmitter


def _matched_demap(field_1d, tx) -> np.ndarray:
    """Single-stream matched filter → bits. No frequency downshift."""
    n_taps = tx.pulse_span_symbols * tx.samples_per_symbol + 1
    rrc = root_raised_cosine(n_taps, tx.samples_per_symbol, tx.roll_off)
    n = field_1d.shape[-1]
    full = np.convolve(field_1d, rrc, mode="full")
    delay = (n_taps - 1) // 2
    filtered = full[delay : delay + n]
    syms = filtered[:: tx.samples_per_symbol]
    return symbols_to_bits(syms, tx.modulation_order)


def test_wdm_transmitter_inherits_from_transmitter():
    """WDMTransmitter is a Transmitter — Protocol-style polymorphism works."""
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=16,
        samples_per_symbol=16,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    assert isinstance(tx, Transmitter)


def test_single_pol_wdm_output_shapes():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        seed=0,
        wdm=cfg,
    )
    sent = tx.generate()
    n_samples = tx.n_symbols * tx.samples_per_symbol
    assert sent.field.shape == (n_samples,)
    assert sent.bits.shape == (3, tx.n_symbols * 2)
    assert sent.symbols.shape == (3, tx.n_symbols)
    assert sent.polarization == "single"
    assert sent.wdm is cfg


def test_dual_pol_wdm_output_shapes():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        seed=0,
        polarization="dual",
        wdm=cfg,
    )
    sent = tx.generate()
    n_samples = tx.n_symbols * tx.samples_per_symbol
    assert sent.field.shape == (2, n_samples)
    assert sent.bits.shape == (2, 3, tx.n_symbols * 2)
    assert sent.symbols.shape == (2, 3, tx.n_symbols)
    assert sent.polarization == "dual"


def test_dual_pol_wdm_pol_streams_are_independent():
    """X and Y carry independent bits on every channel (sub-RNGs are
    spawned per (pol, channel))."""
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        seed=0,
        polarization="dual",
        wdm=cfg,
    )
    sent = tx.generate()
    for ch in range(3):
        assert not np.array_equal(sent.bits[0, ch], sent.bits[1, ch])


def test_dual_pol_wdm_channel_streams_are_independent():
    """Within a single polarisation, channel 0 and channel 1 have different
    bits (sub-RNGs per channel)."""
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        seed=0,
        polarization="dual",
        wdm=cfg,
    )
    sent = tx.generate()
    for pol in range(2):
        assert not np.array_equal(sent.bits[pol, 0], sent.bits[pol, 1])


def test_wdm_transmitter_seed_reproducible():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    kw = dict(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        seed=99,
        polarization="dual",
        wdm=cfg,
    )
    a = WDMTransmitter(**kw).generate()
    b = WDMTransmitter(**kw).generate()
    np.testing.assert_array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.field, b.field)


def test_dual_pol_wdm_deterministic_bits_input():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=32,
        samples_per_symbol=16,
        polarization="dual",
        wdm=cfg,
    )
    k = bits_per_symbol(tx.modulation_order)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 3, tx.n_symbols * k), dtype=np.uint8)
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)


def test_dual_pol_wdm_deterministic_symbols_input():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=32,
        samples_per_symbol=16,
        polarization="dual",
        wdm=cfg,
    )
    rng = np.random.default_rng(0)
    symbols = (
        rng.standard_normal((2, 3, tx.n_symbols)) + 1j * rng.standard_normal((2, 3, tx.n_symbols))
    ) / np.sqrt(2)
    sent = tx.generate(symbols=symbols)
    np.testing.assert_allclose(sent.symbols, symbols, atol=1e-12)


def test_dual_pol_wdm_rejects_wrong_input_shapes():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=32,
        samples_per_symbol=16,
        polarization="dual",
        wdm=cfg,
    )
    with pytest.raises(ValueError, match=r"`bits` must have shape \(2, 3,"):
        tx.generate(bits=np.zeros((3, tx.n_symbols * 2), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"`symbols` must have shape \(2, 3,"):
        tx.generate(symbols=np.zeros((3, tx.n_symbols), dtype=np.complex128))


def test_dual_pol_wdm_b2b_per_channel_per_pol_recovers_bits():
    """Manually downshift each (pol, channel) pair and demap; expect bit-exact
    recovery in the steady-state region. Catches per-pol or per-channel
    seed/scrambling bugs in the new orchestration.
    """
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=1024,
        samples_per_symbol=16,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=7,
        polarization="dual",
        wdm=cfg,
    )
    sent = tx.generate()
    n_samples = sent.field.shape[-1]
    t = np.arange(n_samples) * sent.dt_s
    freqs = cfg.channel_frequencies_hz()
    skip = tx.pulse_span_symbols
    k = bits_per_symbol(tx.modulation_order)

    for pol in (0, 1):
        for ch in range(cfg.n_channels):
            downshifted = sent.field[pol] * np.exp(-2j * np.pi * float(freqs[ch]) * t)
            rx_bits = _matched_demap(downshifted, tx)
            sent_bits = sent.bits[pol, ch].reshape(-1, k)[skip:-skip]
            rx_bits_view = rx_bits.reshape(-1, k)[skip:-skip]
            assert int(np.sum(sent_bits != rx_bits_view)) == 0, (
                f"pol={pol} ch={ch}: bit errors after dual-pol × WDM round-trip"
            )


def test_wdm_transmitter_requires_wdm_field():
    """`wdm` is required — constructing without raises ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="wdm"):
        WDMTransmitter(  # type: ignore[call-arg]
            modulation_order=4,
            symbol_rate_hz=10e9,
            n_symbols=16,
            samples_per_symbol=16,
        )


def test_wdm_transmitter_pilots_on_every_channel_and_pol():
    """PilotConfig applied to a WDMTransmitter writes pilots at the same
    positions across every (pol, channel) pair."""
    cfg_wdm = WDMConfig(n_channels=3, spacing_hz=50e9)
    cfg_p = PilotConfig(period=8, symbol=0.3 - 0.4j)
    tx = WDMTransmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=16,
        pulse_span_symbols=16,
        seed=0,
        polarization="dual",
        wdm=cfg_wdm,
        pilots=cfg_p,
    )
    sent = tx.generate()
    pilot_idx = cfg_p.pilot_indices(tx.n_symbols)
    for pol in range(2):
        for ch in range(cfg_wdm.n_channels):
            np.testing.assert_allclose(sent.symbols[pol, ch, pilot_idx], cfg_p.symbol, atol=1e-12)
