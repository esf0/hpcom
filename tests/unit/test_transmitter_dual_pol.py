"""Dual-polarisation Transmitter generation.

Polarisation-multiplexed (PDM) QAM: independent X and Y bit streams carried on
orthogonal polarisations. Field shape becomes ``(2, n_samples)``; bits and
symbols carry a leading polarisation axis. Today this works for the
single-carrier (no-WDM) path; ``polarization="dual"`` × WDM is reserved for
the upcoming ``WDMTransmitter`` subclass and raises ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import bits_per_symbol
from hpcom.pilots import PilotConfig
from hpcom.transmitter import Transmitter
from hpcom.wdm import WDMConfig


def test_dual_pol_field_shape_and_independence():
    """Dual-pol output is (2, n_samples); X and Y bit streams differ for the
    same root seed (sub-RNGs are independent)."""
    tx = Transmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=256,
        samples_per_symbol=8,
        seed=0,
        polarization="dual",
    )
    sent = tx.generate()
    k = bits_per_symbol(tx.modulation_order)
    assert sent.field.shape == (2, tx.n_symbols * tx.samples_per_symbol)
    assert sent.bits.shape == (2, tx.n_symbols * k)
    assert sent.symbols.shape == (2, tx.n_symbols)
    assert sent.polarization == "dual"
    # X and Y carry independent bits → not identical for nontrivial n.
    assert not np.array_equal(sent.bits[0], sent.bits[1])


def test_dual_pol_seed_deterministic():
    """Same seed → identical output across runs."""
    kwargs = dict(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=8,
        seed=42,
        polarization="dual",
    )
    a = Transmitter(**kwargs).generate()
    b = Transmitter(**kwargs).generate()
    np.testing.assert_array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.field, b.field)


def test_dual_pol_b2b_each_pol_recovers_its_own_bits():
    """Receiver demaps each polarisation independently (test does it via
    direct matched filter call, since the Receiver dual-pol path lands in
    the next commit)."""
    from hpcom._pulse import root_raised_cosine
    from hpcom.modulation import symbols_to_bits

    tx = Transmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=512,
        samples_per_symbol=8,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=1,
        polarization="dual",
    )
    sent = tx.generate()
    n_taps = tx.pulse_span_symbols * tx.samples_per_symbol + 1
    rrc = root_raised_cosine(n_taps, tx.samples_per_symbol, tx.roll_off)
    skip = tx.pulse_span_symbols
    k = bits_per_symbol(tx.modulation_order)
    for pol in (0, 1):
        full = np.convolve(sent.field[pol], rrc, mode="full")
        delay = (n_taps - 1) // 2
        filtered = full[delay : delay + sent.field.shape[-1]]
        rx_syms = filtered[:: tx.samples_per_symbol]
        rx_bits = symbols_to_bits(rx_syms, tx.modulation_order)
        sent_bits = sent.bits[pol].reshape(-1, k)[skip:-skip]
        rx_bits_view = rx_bits.reshape(-1, k)[skip:-skip]
        assert int(np.sum(sent_bits != rx_bits_view)) == 0, (
            f"pol={pol}: dual-pol B2B BER must be 0"
        )


def test_dual_pol_deterministic_bits_input():
    """Feed shape (2, n_bits) into generate(bits=...) — round-trip exactly."""
    n = 64
    k = 2
    bits = np.array(
        [
            np.tile([0, 1], n),  # X: alternating
            np.tile([1, 0], n),  # Y: opposite pattern
        ],
        dtype=np.uint8,
    )
    assert bits.shape == (2, n * k)
    tx = Transmitter(
        modulation_order=4,
        symbol_rate_hz=1e9,
        n_symbols=n,
        polarization="dual",
    )
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.bits, bits)


def test_dual_pol_deterministic_symbols_input():
    n = 32
    symbols = np.zeros((2, n), dtype=np.complex128)
    symbols[0] = 1 + 0j
    symbols[1] = 0 + 1j
    tx = Transmitter(
        modulation_order=4,
        symbol_rate_hz=1e9,
        n_symbols=n,
        polarization="dual",
    )
    sent = tx.generate(symbols=symbols)
    np.testing.assert_allclose(sent.symbols, symbols, atol=1e-12)


def test_dual_pol_pilots_apply_to_both_pols():
    """Same PilotConfig applied to each polarisation: pilot positions carry the
    pilot value on BOTH X and Y."""
    cfg = PilotConfig(period=8, symbol=0.7 + 0.3j)
    tx = Transmitter(
        modulation_order=4,
        symbol_rate_hz=10e9,
        n_symbols=64,
        samples_per_symbol=8,
        pulse_span_symbols=16,
        seed=0,
        polarization="dual",
        pilots=cfg,
    )
    sent = tx.generate()
    pilot_idx = cfg.pilot_indices(tx.n_symbols)
    for pol in (0, 1):
        np.testing.assert_allclose(
            sent.symbols[pol, pilot_idx], cfg.symbol, atol=1e-12
        )


def test_dual_pol_rejects_wrong_input_shapes():
    tx = Transmitter(
        modulation_order=4,
        symbol_rate_hz=1e9,
        n_symbols=10,
        polarization="dual",
    )
    with pytest.raises(ValueError, match=r"dual-pol `bits`"):
        tx.generate(bits=np.zeros(20, dtype=np.uint8))  # 1-D, should be (2, 20)
    with pytest.raises(ValueError, match=r"dual-pol `symbols`"):
        tx.generate(symbols=np.zeros(10, dtype=np.complex128))


def test_dual_pol_plus_wdm_raises_not_implemented():
    """Dual-pol × WDM is deferred to WDMTransmitter (next refactor commit)."""
    with pytest.raises(NotImplementedError, match="WDMTransmitter"):
        Transmitter(
            modulation_order=4,
            symbol_rate_hz=10e9,
            n_symbols=64,
            samples_per_symbol=16,
            polarization="dual",
            wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        )


def test_dual_pol_signal_average_power_is_split_evenly():
    """Without explicit power scaling, both pols emit roughly the same average
    power (unit-power constellation / sps). Real systems would scale by 1/√2
    per pol; that is a Simulation-level concern, not a Tx concern."""
    tx = Transmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=4096,
        samples_per_symbol=8,
        pulse_span_symbols=32,
        seed=0,
        polarization="dual",
    )
    sent = tx.generate()
    px = float(np.mean(np.abs(sent.field[0]) ** 2))
    py = float(np.mean(np.abs(sent.field[1]) ** 2))
    np.testing.assert_allclose(px, py, rtol=0.05)


def test_polarization_invalid_value_rejected():
    with pytest.raises(ValueError, match="polarization"):
        Transmitter(
            modulation_order=4,
            symbol_rate_hz=1e9,
            n_symbols=4,
            polarization="triple",  # type: ignore[arg-type]
        )
