"""Tests for hpcom.modulation."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import (
    bits_per_symbol,
    bits_to_symbols,
    nearest_symbol_indices,
    square_qam,
    symbols_to_bits,
)


@pytest.mark.parametrize("order, k", [(4, 2), (16, 4), (64, 6), (256, 8), (1024, 10)])
def test_bits_per_symbol_matches_log2(order: int, k: int):
    assert bits_per_symbol(order) == k


@pytest.mark.parametrize("order", [3, 0, 6])
def test_bits_per_symbol_rejects_non_power_of_two(order: int):
    with pytest.raises(ValueError):
        bits_per_symbol(order)


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_constellation_has_unit_average_power(order: int):
    constellation, _ = square_qam(order)
    avg_power = float(np.mean(np.abs(constellation) ** 2))
    np.testing.assert_allclose(avg_power, 1.0, atol=1e-12)


@pytest.mark.parametrize("labeling", ["q_descending", "q_ascending"])
@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_bit_labels_unique_per_symbol(order: int, labeling):
    _, bit_labels = square_qam(order, labeling)
    # Each symbol has a unique bit pattern
    seen = {tuple(row) for row in bit_labels}
    assert len(seen) == order


@pytest.mark.parametrize("labeling", ["q_descending", "q_ascending"])
@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_bit_labels_gray_along_each_axis(order: int, labeling):
    """Adjacent constellation points along I or Q must differ by exactly one bit."""
    _, bit_labels = square_qam(order, labeling)
    side = round(float(np.sqrt(order)))
    # Symbols are laid out in a row-major (Q, I) grid.
    for q in range(side):
        for i in range(side - 1):
            here = bit_labels[q * side + i]
            right = bit_labels[q * side + i + 1]
            assert int(np.sum(here != right)) == 1
    for q in range(side - 1):
        for i in range(side):
            here = bit_labels[q * side + i]
            up = bit_labels[(q + 1) * side + i]
            assert int(np.sum(here != up)) == 1


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_round_trip_bits_to_symbols_to_bits(order: int):
    rng = np.random.default_rng(0)
    n_symbols = 1024
    k = bits_per_symbol(order)
    bits = rng.integers(0, 2, size=n_symbols * k, dtype=np.uint8)
    symbols = bits_to_symbols(bits, order)
    recovered = symbols_to_bits(symbols, order)
    np.testing.assert_array_equal(recovered, bits)


def test_nearest_symbol_picks_closest():
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)
    received = np.array([0.9 + 0.8j, -0.7 - 0.7j], dtype=np.complex128)
    idx = nearest_symbol_indices(received, constellation)
    np.testing.assert_array_equal(idx, [0, 2])


def test_bits_length_must_be_multiple_of_k():
    with pytest.raises(ValueError, match="not a multiple"):
        bits_to_symbols(np.array([1, 0, 1], dtype=np.uint8), 16)
