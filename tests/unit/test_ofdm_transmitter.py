"""T12.2: OFDMTransmitter construction, signal structure, and shape contract.

Verifies that ``OFDMTransmitter`` produces a field consistent with the OFDM
specification: IFFT-based modulation with cyclic prefix, guard subcarriers
forced to zero in the frequency domain, pilot subcarriers carrying the
configured pilot value, and power normalisation that puts the data-driven
portion of the field at unit average sample power.

Receive-side correctness (BER=0 noiseless B2B) lives with the
:class:`OFDMReceiver` tests added in T12.3 — these tests only exercise the
Tx side, using manual FFT-based inspection of the generated field.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import (
    ConstantCompositionDM,
    ConstellationConfig,
    maxwell_boltzmann,
    square_qam,
)
from hpcom.ofdm import OFDMConfig, OFDMTransmitter
from hpcom.pilots import PilotConfig
from hpcom.transmitter import Transmitter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tx(
    *,
    n_subcarriers: int = 64,
    cp_len: int = 16,
    n_guard: int = 4,
    pilot_indices: tuple[int, ...] = (),
    n_frames: int = 32,
    modulation_order: int = 4,
    polarization: str = "single",
    seed: int = 0,
) -> OFDMTransmitter:
    ofdm = OFDMConfig(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        n_guard=n_guard,
        pilot_subcarrier_indices=pilot_indices,
    )
    return OFDMTransmitter(
        modulation_order=modulation_order,
        symbol_rate_hz=32e9,
        n_symbols=n_frames,
        ofdm=ofdm,
        polarization=polarization,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Inheritance and metadata
# ---------------------------------------------------------------------------


def test_ofdm_transmitter_is_a_transmitter():
    """Polymorphism: an OFDMTransmitter satisfies the Transmitter API."""
    tx = _make_tx()
    assert isinstance(tx, Transmitter)


def test_dt_s_matches_inverse_bandwidth():
    """D1 — dt_s = 1/symbol_rate_hz (OFDM symbol_rate_hz = total bandwidth)."""
    tx = _make_tx()
    assert tx.dt_s == pytest.approx(1.0 / tx.symbol_rate_hz, rel=1e-15)


def test_ofdm_transmitter_has_no_rrc_fields():
    """OFDM subclasses Transmitter directly (not SingleCarrierTransmitter), so
    it does NOT carry samples_per_symbol / roll_off / pulse_span_symbols."""
    tx = _make_tx(n_subcarriers=64, cp_len=16)
    fields = OFDMTransmitter.model_fields  # class-level, pydantic v2.11+ style
    assert "samples_per_symbol" not in fields
    assert "roll_off" not in fields
    assert "pulse_span_symbols" not in fields
    # Doubly check: OFDM Tx does not store them via __dict__ either.
    assert "samples_per_symbol" not in tx.__dict__


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_base_pilots():
    """Base-class PilotConfig is for time-domain SC pilots; OFDM uses
    frequency-domain pilots via OFDMConfig instead."""
    with pytest.raises(ValueError, match="OFDMTransmitter does not accept"):
        OFDMTransmitter(
            modulation_order=4,
            symbol_rate_hz=32e9,
            n_symbols=10,
            ofdm=OFDMConfig(n_subcarriers=64),
            pilots=PilotConfig(symbol=1.0 + 0j, period=8),
        )


def test_rejects_zero_data_subcarriers():
    """An OFDMConfig with n_data == 0 carries no data — reject up front."""
    # N=8, n_guard=1 leaves indices {0,1,2,5,6,7} as non-guard.
    # Marking all 6 as pilots leaves n_data = 0.
    ofdm = OFDMConfig(
        n_subcarriers=8,
        n_guard=1,
        pilot_subcarrier_indices=(0, 1, 2, 5, 6, 7),
    )
    with pytest.raises(ValueError, match="n_data == 0"):
        OFDMTransmitter(
            modulation_order=4,
            symbol_rate_hz=32e9,
            n_symbols=10,
            ofdm=ofdm,
        )


def test_dm_requires_block_alignment_on_data_symbols():
    """DM block-length divides n_symbols * n_data (total QAM data symbols),
    not just n_symbols."""
    points, _ = square_qam(16)
    pmf = maxwell_boltzmann(points, nu=0.5)
    cfg = ConstellationConfig.from_points(points, probabilities=pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    # N=64, n_guard=4 → n_data=56. n_symbols=3 → total = 168.
    # 168 % 32 = 8 != 0 → must reject.
    ofdm = OFDMConfig(n_subcarriers=64, n_guard=4)
    with pytest.raises(ValueError, match="must be a multiple of"):
        OFDMTransmitter(
            constellation=cfg,
            dm=dm,
            symbol_rate_hz=32e9,
            n_symbols=3,
            ofdm=ofdm,
        )
    # n_symbols=4 → total = 224. 224 % 32 = 0 → accepted.
    tx = OFDMTransmitter(
        constellation=cfg,
        dm=dm,
        symbol_rate_hz=32e9,
        n_symbols=4,
        ofdm=ofdm,
    )
    assert tx.n_symbols == 4


# ---------------------------------------------------------------------------
# Output shapes (single-pol)
# ---------------------------------------------------------------------------


def test_single_pol_field_shape():
    tx = _make_tx(n_subcarriers=64, cp_len=16, n_frames=10)
    sent = tx.generate()
    assert sent.field.shape == (10 * 80,)  # n_frames * (N + cp_len)


def test_single_pol_symbols_shape():
    tx = _make_tx(n_subcarriers=64, n_guard=4, n_frames=10)
    sent = tx.generate()
    # n_data = 64 - 8 = 56; symbols are flat (n_frames * n_data,)
    assert sent.symbols.shape == (10 * 56,)


def test_single_pol_bits_shape_qpsk():
    tx = _make_tx(modulation_order=4, n_subcarriers=64, n_guard=4, n_frames=10)
    sent = tx.generate()
    # 10 * 56 symbols * 2 bits/symbol (QPSK)
    assert sent.bits.shape == (10 * 56 * 2,)


def test_single_pol_bits_shape_64qam():
    tx = _make_tx(modulation_order=64, n_subcarriers=64, n_guard=4, n_frames=4)
    sent = tx.generate()
    # 4 * 56 symbols * 6 bits/symbol (64-QAM)
    assert sent.bits.shape == (4 * 56 * 6,)


# ---------------------------------------------------------------------------
# Output shapes (dual-pol)
# ---------------------------------------------------------------------------


def test_dual_pol_field_shape():
    tx = _make_tx(polarization="dual", n_frames=10)
    sent = tx.generate()
    assert sent.field.shape == (2, 10 * 80)


def test_dual_pol_symbols_and_bits_shape():
    tx = _make_tx(polarization="dual", modulation_order=16, n_frames=8)
    sent = tx.generate()
    # n_data = 56, k = 4 (16-QAM)
    assert sent.symbols.shape == (2, 8 * 56)
    assert sent.bits.shape == (2, 8 * 56 * 4)


def test_dual_pol_streams_are_independent():
    """X and Y polarisations are seeded independently — bits should differ."""
    tx = _make_tx(polarization="dual", n_frames=8)
    sent = tx.generate()
    # Statistically: probability of identical 8*56*2 = 896 bits across 2 pols
    # is 2^-896. Just check they are not byte-identical.
    assert not np.array_equal(sent.bits[0], sent.bits[1])


# ---------------------------------------------------------------------------
# Cyclic prefix invariant
# ---------------------------------------------------------------------------


def test_cyclic_prefix_duplicates_tail():
    """The first cp_len samples of each frame must equal the last cp_len
    samples of that frame's IFFT block (the CP is the tail of the
    N-sample useful interval, prepended). This is the OFDM identity that
    makes the linear convolution with a CP-shorter channel look circular.
    """
    tx = _make_tx(n_subcarriers=64, cp_len=16, n_frames=4)
    sent = tx.generate()
    sps = 80  # 64 + 16
    for frame in range(4):
        block = sent.field[frame * sps : (frame + 1) * sps]
        cp_part = block[:16]
        tail = block[-16:]  # last 16 samples of the (cp + ifft) block
        np.testing.assert_allclose(cp_part, tail, atol=1e-12)


def test_no_cyclic_prefix_when_cp_zero():
    """cp_len=0 means no CP is prepended; frame length == n_subcarriers."""
    tx = _make_tx(n_subcarriers=64, cp_len=0, n_frames=4)
    sent = tx.generate()
    assert sent.field.shape == (4 * 64,)


# ---------------------------------------------------------------------------
# Frequency-domain inspection
# ---------------------------------------------------------------------------


def test_guard_subcarriers_zero_in_fft():
    """Guard subcarriers must be zero in the frequency domain of each frame
    (after CP removal, before any noise)."""
    tx = _make_tx(n_subcarriers=64, cp_len=16, n_guard=4, n_frames=4)
    sent = tx.generate()
    sps = 80
    guard_idx = tx.ofdm.guard_subcarrier_indices()
    for frame in range(4):
        block = sent.field[frame * sps : (frame + 1) * sps]
        useful = block[16:]  # strip CP
        spectrum = np.fft.fft(useful)
        # The Tx scaled time-domain by N/sqrt(n_data); FFT of that gives
        # N * (N/sqrt(n_data)) * X_freq, where X_freq is the original
        # frequency-domain input. Guard entries were zero in X_freq.
        np.testing.assert_allclose(
            spectrum[guard_idx],
            0.0,
            atol=1e-9,
            err_msg=f"frame {frame}",
        )


def test_pilot_subcarriers_carry_configured_value():
    """After CP removal + FFT + power de-normalisation, pilot subcarriers
    must carry exactly the configured pilot symbol."""
    tx = _make_tx(
        n_subcarriers=64,
        cp_len=16,
        n_guard=4,
        pilot_indices=(5, 11, 25),
        n_frames=4,
    )
    sent = tx.generate()
    sps = 80
    n = 64
    n_data = tx.ofdm.n_data
    pilot_idx = list(tx.ofdm.pilot_subcarrier_indices)
    expected_pilot = tx.ofdm.pilot_symbol
    # Tx scale = N / sqrt(n_data); to recover X_freq from FFT(time-domain):
    # FFT(time) = N * X_freq * (N / sqrt(n_data)) / N = X_freq * N / sqrt(n_data)
    # so X_freq = FFT(time) * sqrt(n_data) / N.
    rescale = np.sqrt(n_data) / n
    for frame in range(4):
        block = sent.field[frame * sps : (frame + 1) * sps]
        useful = block[16:]
        spectrum = np.fft.fft(useful) * rescale
        np.testing.assert_allclose(
            spectrum[pilot_idx],
            expected_pilot,
            atol=1e-10,
            err_msg=f"frame {frame}",
        )


def test_data_subcarriers_round_trip_to_qam_symbols():
    """The transmitted ``symbols`` field equals what's at the data subcarriers
    in the frequency domain (modulo the Tx power normalisation)."""
    tx = _make_tx(n_subcarriers=64, cp_len=16, n_guard=4, n_frames=4)
    sent = tx.generate()
    sps = 80
    n = 64
    n_data = tx.ofdm.n_data
    data_idx = tx.ofdm.data_subcarrier_indices()
    rescale = np.sqrt(n_data) / n
    symbols_per_frame = sent.symbols.reshape(4, n_data)
    for frame in range(4):
        block = sent.field[frame * sps : (frame + 1) * sps]
        useful = block[16:]
        spectrum = np.fft.fft(useful) * rescale
        np.testing.assert_allclose(
            spectrum[data_idx],
            symbols_per_frame[frame],
            atol=1e-10,
            err_msg=f"frame {frame}",
        )


# ---------------------------------------------------------------------------
# Power normalisation
# ---------------------------------------------------------------------------


def test_unit_average_power_without_pilots():
    """With unit-power constellation and no pilots, the field has average
    sample power ≈ 1 (D4 power norm). Statistical tolerance from many frames."""
    tx = _make_tx(n_subcarriers=128, cp_len=16, n_guard=4, n_frames=200, seed=42)
    sent = tx.generate()
    avg_power = float(np.mean(np.abs(sent.field) ** 2))
    # With 200 frames × n_data data symbols ≈ 23200 random symbols,
    # std of avg power ≈ 1/sqrt(N_samples) ≈ 0.007. Use 0.02 for headroom.
    assert avg_power == pytest.approx(1.0, abs=0.02)


def test_average_power_above_one_with_pilots():
    """Pilots add deterministic energy; field power = 1 + n_pilots/n_data
    (with |pilot_symbol|² = 1)."""
    pilot_idx = (3, 5, 7, 9)  # 4 pilots
    tx = _make_tx(
        n_subcarriers=128,
        cp_len=16,
        n_guard=4,
        pilot_indices=pilot_idx,
        n_frames=200,
        seed=42,
    )
    sent = tx.generate()
    n_data = tx.ofdm.n_data  # 128 - 8 - 4 = 116
    expected = 1.0 + len(pilot_idx) / n_data
    avg_power = float(np.mean(np.abs(sent.field) ** 2))
    assert avg_power == pytest.approx(expected, abs=0.03)


# ---------------------------------------------------------------------------
# Deterministic input paths
# ---------------------------------------------------------------------------


def test_deterministic_bits_input_round_trip():
    """Caller-supplied bits flow through and appear in sent.bits unchanged."""
    tx = _make_tx(n_subcarriers=64, n_guard=4, n_frames=4)
    n_bits = tx._bits_per_stream()
    rng = np.random.default_rng(7)
    user_bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    sent = tx.generate(bits=user_bits)
    np.testing.assert_array_equal(sent.bits, user_bits)


def test_deterministic_symbols_input_round_trip():
    """Caller-supplied symbols flow through and appear in sent.symbols
    unchanged; corresponding bits are recovered via demap."""
    tx = _make_tx(n_subcarriers=64, n_guard=4, n_frames=4)
    cfg = tx._resolved_constellation()
    rng = np.random.default_rng(8)
    n_total = tx._data_symbols_per_stream()
    idx = rng.integers(0, cfg.modulation_order, size=n_total)
    user_syms = cfg.points[idx]
    sent = tx.generate(symbols=user_syms)
    np.testing.assert_allclose(sent.symbols, user_syms, atol=1e-12)


def test_rejects_wrong_bits_shape():
    tx = _make_tx(n_frames=4)
    with pytest.raises(ValueError, match="shape"):
        tx.generate(bits=np.zeros(10, dtype=np.uint8))


def test_rejects_wrong_symbols_shape():
    tx = _make_tx(n_frames=4)
    with pytest.raises(ValueError, match="shape"):
        tx.generate(symbols=np.zeros(10, dtype=np.complex128))


def test_rejects_both_bits_and_symbols():
    tx = _make_tx(n_frames=4)
    with pytest.raises(ValueError, match="at most one"):
        tx.generate(
            bits=np.zeros(tx._bits_per_stream(), dtype=np.uint8),
            symbols=np.zeros(tx._data_symbols_per_stream(), dtype=np.complex128),
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_output():
    tx_a = _make_tx(seed=123)
    tx_b = _make_tx(seed=123)
    sent_a = tx_a.generate()
    sent_b = tx_b.generate()
    np.testing.assert_array_equal(sent_a.bits, sent_b.bits)
    np.testing.assert_allclose(sent_a.field, sent_b.field, atol=1e-15)


def test_different_seed_produces_different_output():
    tx_a = _make_tx(seed=1)
    tx_b = _make_tx(seed=2)
    sent_a = tx_a.generate()
    sent_b = tx_b.generate()
    assert not np.array_equal(sent_a.bits, sent_b.bits)


# ---------------------------------------------------------------------------
# DM (probabilistic shaping) integration
# ---------------------------------------------------------------------------


def test_dm_path_emits_correct_bit_count():
    """With DM, sent.bits has DM-sized length (not n_data_syms * k)."""
    points, _ = square_qam(16)
    pmf = maxwell_boltzmann(points, nu=0.5)
    cfg = ConstellationConfig.from_points(points, probabilities=pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    # N=64, n_guard=4 → n_data=56. n_frames=4 → 224 total = 7 DM blocks.
    ofdm = OFDMConfig(n_subcarriers=64, n_guard=4)
    tx = OFDMTransmitter(
        constellation=cfg,
        dm=dm,
        symbol_rate_hz=32e9,
        n_symbols=4,
        ofdm=ofdm,
        seed=0,
    )
    sent = tx.generate()
    expected = 7 * dm.bits_per_block
    assert sent.bits.shape == (expected,)


def test_dm_path_empirical_pmf_matches_composition():
    """The transmitted symbols' empirical distribution equals the DM
    composition (one full DM block emits exactly the composition once)."""
    points, _ = square_qam(16)
    pmf = maxwell_boltzmann(points, nu=0.5)
    cfg = ConstellationConfig.from_points(points, probabilities=pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    ofdm = OFDMConfig(n_subcarriers=64, n_guard=4)
    # 16 frames × 56 data subs = 896 = 28 DM blocks.
    tx = OFDMTransmitter(
        constellation=cfg,
        dm=dm,
        symbol_rate_hz=32e9,
        n_symbols=16,
        ofdm=ofdm,
        seed=0,
    )
    sent = tx.generate()
    # Find nearest constellation index for each sent symbol.
    diffs = sent.symbols[:, None] - cfg.points[None, :]
    indices = np.argmin(np.abs(diffs), axis=1)
    counts = np.bincount(indices, minlength=cfg.modulation_order)
    empirical = counts / indices.size
    np.testing.assert_allclose(empirical, dm.empirical_pmf, atol=1e-12)
