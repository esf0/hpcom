"""End-to-end probabilistic shaping: DM wired through Transmitter/Receiver.

Verifies the dm= keyword on Transmitter and Receiver. With dm set, the
input bit stream is run through the CC-DM to produce shaped symbol
indices, and the receiver's nearest-neighbour decisions are run back
through dm.decode to recover bits.
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
from hpcom.pilots import PilotConfig
from hpcom.receiver import Receiver
from hpcom.transmitter import Transmitter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_8psk_shaped() -> tuple[ConstellationConfig, ConstantCompositionDM]:
    """8-PSK + M-B(ν=0.6): equi-energy points → M-B reduces to uniform; the
    DM still operates over an explicit (uniform) probability vector, which
    exercises the API end-to-end."""
    points = np.exp(2j * np.pi * np.arange(8) / 8)
    pmf = maxwell_boltzmann(points, nu=0.6)
    cfg = ConstellationConfig.from_points(points, probabilities=pmf)
    dm = ConstantCompositionDM(pmf, block_length=32)
    return cfg, dm


def _make_16qam_shaped() -> tuple[ConstellationConfig, ConstantCompositionDM]:
    """16-QAM + M-B(ν=0.6): non-uniform energies → actually shaped PMF."""
    points, _ = square_qam(16)
    pmf = maxwell_boltzmann(points, nu=0.6)
    cfg = ConstellationConfig.from_points(points, probabilities=pmf)
    dm = ConstantCompositionDM(pmf, block_length=32)
    return cfg, dm


@pytest.fixture(params=["8psk", "16qam"])
def shaped_cfg_and_dm(request):
    """Parametrised fixture: every consumer test runs against both 8-PSK
    (equi-energy, M-B degenerates to uniform) and 16-QAM (truly shaped)."""
    return _make_8psk_shaped() if request.param == "8psk" else _make_16qam_shaped()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_tx_dm_requires_constellation_with_probabilities():
    """Passing dm without a probabilistic constellation must raise."""
    points = np.exp(2j * np.pi * np.arange(8) / 8)
    cfg_unshaped = ConstellationConfig.from_points(points)  # uniform
    pmf = maxwell_boltzmann(points, nu=0.5)
    dm = ConstantCompositionDM(pmf, block_length=32)
    with pytest.raises(ValueError, match="probabilities"):
        Transmitter(
            constellation=cfg_unshaped, dm=dm,
            symbol_rate_hz=32e9, n_symbols=256,
        )


def test_tx_dm_target_pmf_must_match_constellation(shaped_cfg_and_dm):
    cfg, _ = shaped_cfg_and_dm
    # 8-PSK has equal-energy points so M-B is always uniform on it. Construct
    # an arbitrary unrelated PMF that differs from cfg.probabilities.
    other_pmf = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
    other_dm = ConstantCompositionDM(other_pmf, block_length=32)
    with pytest.raises(ValueError, match="target_pmf"):
        Transmitter(
            constellation=cfg, dm=other_dm,
            symbol_rate_hz=32e9, n_symbols=256,
        )


def test_tx_n_symbols_must_be_multiple_of_block_length(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm  # block_length=32
    with pytest.raises(ValueError, match="multiple of"):
        Transmitter(
            constellation=cfg, dm=dm,
            symbol_rate_hz=32e9, n_symbols=257,
        )


def test_tx_pilots_and_dm_mutually_exclusive(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm
    pilots = PilotConfig(symbol=1.0 + 0j, period=8)
    with pytest.raises(ValueError, match=r"pilots.*dm"):
        Transmitter(
            constellation=cfg, dm=dm, pilots=pilots,
            symbol_rate_hz=32e9, n_symbols=256,
        )


# ---------------------------------------------------------------------------
# Empirical PMF on transmitted symbols matches the DM composition
# ---------------------------------------------------------------------------

def test_transmitted_symbols_follow_dm_composition(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm
    tx = Transmitter(
        constellation=cfg, dm=dm,
        symbol_rate_hz=32e9, n_symbols=32 * 8,  # 8 DM blocks
        samples_per_symbol=8, pulse_span_symbols=16, seed=0,
    )
    sent = tx.generate()

    # Recover symbol indices via nearest-neighbour on the noiseless symbols
    # (symbols are exactly on the constellation since map() is a lookup).
    diffs = sent.symbols[:, None] - cfg.points[None, :]
    indices = np.argmin(np.abs(diffs), axis=1)
    counts = np.bincount(indices, minlength=cfg.modulation_order)
    empirical = counts / indices.size

    # Should match composition/N exactly (each block emits the composition).
    np.testing.assert_allclose(empirical, dm.empirical_pmf, atol=1e-12)


# ---------------------------------------------------------------------------
# Bit-stream length
# ---------------------------------------------------------------------------

def test_tx_emits_dm_sized_bit_stream(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm
    tx = Transmitter(
        constellation=cfg, dm=dm,
        symbol_rate_hz=32e9, n_symbols=32 * 4,
        seed=1,
    )
    sent = tx.generate()
    expected = (tx.n_symbols // dm.block_length) * dm.bits_per_block
    assert sent.bits.shape == (expected,)


# ---------------------------------------------------------------------------
# B2B round-trip (noiseless)
# ---------------------------------------------------------------------------

def test_b2b_round_trip_with_dm(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm
    span = 16
    tx = Transmitter(
        constellation=cfg, dm=dm,
        symbol_rate_hz=32e9, n_symbols=32 * 16,
        samples_per_symbol=8, pulse_span_symbols=span, seed=2,
    )
    rx = Receiver(
        constellation=cfg, dm=dm,
        samples_per_symbol=8, pulse_span_symbols=span,
    )

    sent = tx.generate()
    received = rx.receive(sent.field)

    # RRC truncation at the edges flips a few outer-ring symbols, which
    # propagates a whole block of DM-decoded bits per affected block. Trim
    # block-aligned by dropping the first/last block at the symbol level
    # before decoding via the DM is overkill; instead trim the bits view
    # using the bit↔block correspondence.
    bits_per_block = dm.bits_per_block
    skip_blocks = max(1, span // dm.block_length + 1)
    sent_trim = sent.bits[skip_blocks * bits_per_block : -skip_blocks * bits_per_block]
    rx_trim = received.bits[skip_blocks * bits_per_block : -skip_blocks * bits_per_block]
    np.testing.assert_array_equal(rx_trim, sent_trim)


def test_b2b_round_trip_with_dm_dual_pol(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm
    span = 16
    tx = Transmitter(
        constellation=cfg, dm=dm,
        symbol_rate_hz=32e9, n_symbols=32 * 16,
        samples_per_symbol=8, pulse_span_symbols=span,
        polarization="dual", seed=3,
    )
    rx = Receiver(
        constellation=cfg, dm=dm,
        samples_per_symbol=8, pulse_span_symbols=span,
        polarization="dual",
    )

    sent = tx.generate()
    received = rx.receive(sent.field)

    bits_per_block = dm.bits_per_block
    skip_blocks = max(1, span // dm.block_length + 1)
    skip_bits = skip_blocks * bits_per_block
    for pol in (0, 1):
        sent_trim = sent.bits[pol, skip_bits:-skip_bits]
        rx_trim = received.bits[pol, skip_bits:-skip_bits]
        np.testing.assert_array_equal(rx_trim, sent_trim)


# ---------------------------------------------------------------------------
# Deterministic-input path
# ---------------------------------------------------------------------------

def test_dm_deterministic_bit_input(shaped_cfg_and_dm):
    """Caller-supplied bits flow through the DM encoder; output preserves them."""
    cfg, dm = shaped_cfg_and_dm
    tx = Transmitter(
        constellation=cfg, dm=dm,
        symbol_rate_hz=32e9, n_symbols=32 * 4,
        seed=4,
    )
    n_bits = (tx.n_symbols // dm.block_length) * dm.bits_per_block
    rng = np.random.default_rng(99)
    user_bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    sent = tx.generate(bits=user_bits)
    np.testing.assert_array_equal(sent.bits, user_bits)


def test_dm_rejects_wrong_bit_input_length(shaped_cfg_and_dm):
    cfg, dm = shaped_cfg_and_dm
    tx = Transmitter(
        constellation=cfg, dm=dm,
        symbol_rate_hz=32e9, n_symbols=32 * 4,
    )
    wrong = np.zeros(10, dtype=np.uint8)  # wrong size
    with pytest.raises(ValueError, match="shape"):
        tx.generate(bits=wrong)
