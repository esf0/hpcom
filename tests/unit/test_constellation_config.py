"""Unit tests for ConstellationConfig (T11.1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hpcom.modulation import (
    ConstellationConfig,
    bits_to_symbols,
    square_qam,
    symbols_to_bits,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_non_power_of_two_size():
    points = np.array([1 + 0j, -1 + 0j, 0 + 1j], dtype=np.complex128)
    bit_labels = np.zeros((3, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="power-of-2"):
        ConstellationConfig(points=points, bit_labels=bit_labels)


def test_rejects_wrong_bit_label_shape():
    points = np.array([1 + 0j, -1 + 0j], dtype=np.complex128)  # M=2, k=1
    bit_labels = np.zeros((2, 2), dtype=np.uint8)  # wrong k
    with pytest.raises(ValueError, match=r"shape \(2, 1\)"):
        ConstellationConfig(points=points, bit_labels=bit_labels)


def test_rejects_duplicate_bit_labels():
    points = np.array([1 + 0j, -1 + 0j, 1j, -1j], dtype=np.complex128) / np.sqrt(1.0)
    points = points / np.sqrt(np.mean(np.abs(points) ** 2))
    bit_labels = np.array([[0, 0], [0, 1], [0, 1], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unique"):
        ConstellationConfig(points=points, bit_labels=bit_labels)


def test_rejects_non_binary_bit_labels():
    points, bit_labels = square_qam(4)
    bit_labels = bit_labels.copy()
    bit_labels[0, 0] = 2
    with pytest.raises(ValueError, match="0/1"):
        ConstellationConfig(points=points, bit_labels=bit_labels)


def test_rejects_unnormalised_points():
    points, bit_labels = square_qam(4)
    with pytest.raises(ValueError, match="expected symbol power"):
        ConstellationConfig(points=points * 2.0, bit_labels=bit_labels)


def test_rejects_probabilities_wrong_shape():
    points, bit_labels = square_qam(4)
    bad_probs = np.array([0.25, 0.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        ConstellationConfig(points=points, bit_labels=bit_labels, probabilities=bad_probs)


def test_rejects_probabilities_not_summing_to_one():
    points, bit_labels = square_qam(4)
    bad_probs = np.array([0.25, 0.25, 0.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="sum to 1"):
        ConstellationConfig(points=points, bit_labels=bit_labels, probabilities=bad_probs)


def test_rejects_negative_probabilities():
    points, bit_labels = square_qam(4)
    bad_probs = np.array([-0.1, 0.4, 0.3, 0.4], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        ConstellationConfig(points=points, bit_labels=bit_labels, probabilities=bad_probs)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [4, 16, 64, 256])
def test_from_square_qam_matches_flat_api(m):
    cfg = ConstellationConfig.from_square_qam(m)
    assert cfg.modulation_order == m
    assert cfg.bits_per_symbol == int(np.log2(m))

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=cfg.bits_per_symbol * 1024, dtype=np.uint8)

    symbols_cfg = cfg.map(bits)
    symbols_flat = bits_to_symbols(bits, m)
    np.testing.assert_array_equal(symbols_cfg, symbols_flat)

    bits_back = cfg.demap(symbols_cfg)
    np.testing.assert_array_equal(bits_back, bits)


def test_from_square_qam_demap_matches_flat_api():
    cfg = ConstellationConfig.from_square_qam(16)
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=cfg.bits_per_symbol * 512, dtype=np.uint8)
    symbols = bits_to_symbols(bits, 16)
    # Add small noise so the nearest-neighbour decision is exercised.
    noisy = symbols + 0.05 * (rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape))

    np.testing.assert_array_equal(cfg.demap(noisy), symbols_to_bits(noisy, 16))


def test_from_points_normalises_to_unit_power():
    raw_points = np.array([2 + 0j, -2 + 0j, 0 + 2j, 0 - 2j], dtype=np.complex128)
    cfg = ConstellationConfig.from_points(raw_points)
    power = float(np.mean(np.abs(cfg.points) ** 2))
    assert power == pytest.approx(1.0, abs=1e-12)
    assert cfg.modulation_order == 4
    assert cfg.bits_per_symbol == 2


def test_from_points_default_labels_are_unique():
    raw_points = np.exp(2j * np.pi * np.arange(8) / 8)  # 8-PSK
    cfg = ConstellationConfig.from_points(raw_points)
    assert cfg.bit_labels.shape == (8, 3)
    # Unique rows.
    packed = (cfg.bit_labels.astype(int) * np.array([4, 2, 1])).sum(axis=1)
    assert np.unique(packed).size == 8


def test_from_points_round_trip_with_default_labels():
    raw_points = np.exp(2j * np.pi * np.arange(8) / 8)
    cfg = ConstellationConfig.from_points(raw_points)
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=cfg.bits_per_symbol * 256, dtype=np.uint8)
    symbols = cfg.map(bits)
    # Noiseless demap recovers the input.
    np.testing.assert_array_equal(cfg.demap(symbols), bits)


def test_from_points_with_probabilities_normalises_expected_power():
    raw_points = np.array([3 + 0j, 1 + 0j, -1 + 0j, -3 + 0j], dtype=np.complex128)
    probs = np.array([0.1, 0.4, 0.4, 0.1], dtype=np.float64)
    cfg = ConstellationConfig.from_points(raw_points, probabilities=probs)
    expected = float(np.dot(cfg.probabilities, np.abs(cfg.points) ** 2))
    assert expected == pytest.approx(1.0, abs=1e-12)
    assert not cfg.is_uniform


# ---------------------------------------------------------------------------
# Frozen / immutable
# ---------------------------------------------------------------------------

def test_dataclass_is_frozen():
    cfg = ConstellationConfig.from_square_qam(4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.points = np.zeros(4, dtype=np.complex128)
