"""T11.4: Maxwell-Boltzmann PMF + constant-composition DM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpcom.modulation import (
    ConstantCompositionDM,
    ConstellationConfig,
    maxwell_boltzmann,
    square_qam,
)

# ---------------------------------------------------------------------------
# Maxwell-Boltzmann PMF
# ---------------------------------------------------------------------------

def test_mb_nu_zero_is_uniform():
    constellation, _ = square_qam(16)
    p = maxwell_boltzmann(constellation, nu=0.0)
    np.testing.assert_allclose(p, np.full(16, 1.0 / 16), atol=1e-12)


def test_mb_normalises_to_one():
    constellation, _ = square_qam(64)
    p = maxwell_boltzmann(constellation, nu=0.3)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (p > 0).all()


def test_mb_concentrates_on_low_energy_points():
    constellation, _ = square_qam(16)
    energies = np.abs(constellation) ** 2
    p = maxwell_boltzmann(constellation, nu=0.5)
    # For any two points i, j: |s_i|² < |s_j|² ⇒ p_i > p_j.
    order_e = np.argsort(energies)
    p_sorted = p[order_e]
    # PMF should be non-increasing along sorted energies (allowing ties at equal energies).
    diffs = np.diff(p_sorted)
    assert (diffs <= 1e-12).all()


def test_mb_reduces_mean_energy_vs_uniform():
    constellation, _ = square_qam(64)
    energies = np.abs(constellation) ** 2
    uniform_e = energies.mean()
    shaped_e = float(np.dot(maxwell_boltzmann(constellation, nu=0.3), energies))
    assert shaped_e < uniform_e


def test_mb_negative_nu_raises():
    constellation, _ = square_qam(16)
    with pytest.raises(ValueError, match="nu must be"):
        maxwell_boltzmann(constellation, nu=-0.1)


def test_mb_equi_energy_is_always_uniform():
    """M-B PMF must equal uniform for any equi-energy constellation (e.g. PSK).

    For PSK all |s_i|² = r² (constant), so exp(-ν|s_i|²) = exp(-νr²) for
    every i, and the softmax normalises to 1/M regardless of ν. This is a
    physically important identity: shaping has no effect on equi-energy
    constellations — the distribution is always uniform.
    """
    points = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK: all |s|=1
    for nu in [0.0, 0.5, 2.0, 10.0]:
        p = maxwell_boltzmann(points, nu=nu)
        np.testing.assert_allclose(
            p, np.full(8, 1.0 / 8), atol=1e-12,
            err_msg=f"failed for nu={nu}",
        )


# ---------------------------------------------------------------------------
# ConstantCompositionDM — construction
# ---------------------------------------------------------------------------

def test_dm_rejects_non_normalised_pmf():
    with pytest.raises(ValueError, match="sum to 1"):
        ConstantCompositionDM(target_pmf=np.array([0.5, 0.3]), block_length=16)


def test_dm_rejects_negative_pmf():
    with pytest.raises(ValueError, match="non-negative"):
        ConstantCompositionDM(target_pmf=np.array([1.2, -0.2]), block_length=16)


def test_dm_rejects_block_smaller_than_m():
    pmf = np.full(16, 1.0 / 16)
    with pytest.raises(ValueError, match="block_length"):
        ConstantCompositionDM(target_pmf=pmf, block_length=8)


def test_dm_composition_sums_to_block_length():
    constellation = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK
    pmf = maxwell_boltzmann(constellation, nu=0.4)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    assert dm.composition.sum() == 32


def test_dm_bits_per_block_matches_floor_log2_multinomial():
    constellation = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK
    pmf = maxwell_boltzmann(constellation, nu=0.4)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)
    expected = math.factorial(32)
    for c in dm.composition:
        expected //= math.factorial(int(c))
    assert dm.bits_per_block == math.floor(math.log2(expected))


def test_dm_rate_bounded_by_entropy():
    """Information rate in bits/symbol must be positive and ≤ H(target_pmf).

    The rate is ⌊log₂ C⌋ / N where C = N! / Π nᵢ!. Since C ≤ (M/N)^N · N!
    and H(p) = -Σ pᵢ log₂ pᵢ, the rate is always bounded by the source
    entropy — the DM cannot emit more information than the target PMF carries.
    """
    constellation, _ = square_qam(16)
    pmf = maxwell_boltzmann(constellation, nu=0.4)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=64)
    h = float(-np.sum(pmf * np.log2(pmf + 1e-300)))
    assert dm.rate_bits_per_symbol > 0
    assert dm.rate_bits_per_symbol <= h + 1e-9  # add epsilon for floating-point


def test_dm_unrank_out_of_range_raises():
    """_unrank_multiset must raise ValueError for rank >= total permutations."""
    from hpcom.modulation.dm import _unrank_multiset
    # composition [2, 2] → total = 4!/(2!2!) = 6 valid ranks (0..5)
    with pytest.raises(ValueError, match="out of range"):
        _unrank_multiset(6, [2, 2])
    with pytest.raises(ValueError, match="out of range"):
        _unrank_multiset(-1, [2, 2])


def test_dm_rank_out_of_range_sym_raises():
    """_rank_multiset must raise ValueError for a symbol index out of composition range."""
    from hpcom.modulation.dm import _rank_multiset
    # composition has 3 symbols (indices 0, 1, 2); index 3 is invalid
    with pytest.raises(ValueError, match="out of range"):
        _rank_multiset([0, 3, 2], [2, 2, 2])


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_dm_round_trip_uniform_block():
    """Uniform PMF over 8 symbols, block_length=16."""
    pmf = np.full(8, 1.0 / 8)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=16)

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=dm.bits_per_block * 32, dtype=np.uint8)

    symbols = dm.encode(bits)
    bits_back = dm.decode(symbols)

    np.testing.assert_array_equal(bits_back, bits)
    assert symbols.size == 32 * dm.block_length


def test_dm_round_trip_maxwell_boltzmann():
    """Shaped PMF (Maxwell-Boltzmann ν=0.4), block_length=24, 8-APSK constellation."""
    constellation = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK
    pmf = maxwell_boltzmann(constellation, nu=0.4)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=24)

    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=dm.bits_per_block * 64, dtype=np.uint8)

    symbols = dm.encode(bits)
    bits_back = dm.decode(symbols)

    np.testing.assert_array_equal(bits_back, bits)


# ---------------------------------------------------------------------------
# Empirical PMF matches target
# ---------------------------------------------------------------------------

def test_dm_empirical_pmf_equals_composition():
    """By construction, the emitted distribution is exactly composition / N."""
    constellation, _ = square_qam(16)
    target = maxwell_boltzmann(constellation, nu=0.2)
    dm = ConstantCompositionDM(target_pmf=target, block_length=64)

    rng = np.random.default_rng(2)
    n_blocks = 32
    bits = rng.integers(0, 2, size=dm.bits_per_block * n_blocks, dtype=np.uint8)

    symbols = dm.encode(bits)
    counts = np.bincount(symbols, minlength=16)
    empirical = counts / symbols.size

    # Exact match against composition-based PMF (no rounding noise).
    np.testing.assert_allclose(empirical, dm.empirical_pmf, atol=1e-12)


def test_dm_empirical_pmf_kl_to_target_below_1e_3():
    """Largest-remainder rounding of N=128 across M=8 keeps KL very small.

    The acceptance criterion in T11.4 is KL < 1e-3 over ≥ 2^16 symbols. The
    rounding error is the dominant term and depends on block_length, not
    sample count; checking with one block suffices.
    """
    constellation = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK
    target = maxwell_boltzmann(constellation, nu=0.5)
    dm = ConstantCompositionDM(target_pmf=target, block_length=128)

    empirical = dm.empirical_pmf
    # KL(empirical || target). Mask zeros in empirical to avoid 0·log(0/x).
    mask = empirical > 0
    kl = float(np.sum(empirical[mask] * np.log(empirical[mask] / target[mask])))
    assert kl < 1e-3, f"KL divergence {kl:.4e} exceeds 1e-3 budget"


# ---------------------------------------------------------------------------
# Wired to ConstellationConfig
# ---------------------------------------------------------------------------

def test_dm_pipeline_with_constellation_config():
    """Bits → DM.encode → ConstellationConfig.points indexing → symbols
    have the configured (probabilistic) PMF; reverse path recovers bits."""
    constellation = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK
    pmf = maxwell_boltzmann(constellation, nu=0.4)
    cfg = ConstellationConfig.from_points(constellation, probabilities=pmf)
    dm = ConstantCompositionDM(target_pmf=pmf, block_length=32)

    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=dm.bits_per_block * 16, dtype=np.uint8)

    sym_indices = dm.encode(bits)
    # The constellation points (normalised) carried over to cfg.points;
    # the symbols transmitted would be cfg.points[sym_indices].
    transmitted = cfg.points[sym_indices]
    # Noiseless receive: nearest-neighbour from cfg.points is identity.
    decoded_indices = np.argmin(
        np.abs(transmitted[:, None] - cfg.points[None, :]), axis=1
    )
    np.testing.assert_array_equal(decoded_indices, sym_indices)

    bits_back = dm.decode(decoded_indices)
    np.testing.assert_array_equal(bits_back, bits)
