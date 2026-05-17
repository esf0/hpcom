"""Modulation extras (Test A): fixed mappings, completeness, round-trips.

Complements test_modulation.py with checks that catch a different class of
regression — wrong MSB/LSB convention, missing constellation points, bit
labels reused across symbols, etc.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import bits_per_symbol, bits_to_symbols, symbols_to_bits
from hpcom.modulation.constellations import square_qam


@pytest.mark.parametrize("order", [4, 16, 64, 256, 1024])
def test_constellation_is_zero_mean(order: int):
    """Square-QAM, normalized, must be centered at the origin (no DC bias)."""
    constellation, _ = square_qam(order)
    np.testing.assert_allclose(np.mean(constellation), 0.0, atol=1e-12)


@pytest.mark.parametrize("order", [4, 16, 64, 256, 1024])
def test_constellation_points_are_distinct(order: int):
    constellation, _ = square_qam(order)
    assert len({(float(z.real), float(z.imag)) for z in constellation}) == order


@pytest.mark.parametrize("order", [4, 16, 64, 256, 1024])
def test_every_bit_pattern_appears_exactly_once(order: int):
    """All M unique k-bit patterns must label distinct symbols, no repeats."""
    _, labels = square_qam(order)
    k = bits_per_symbol(order)
    packed = (labels.astype(np.int64) * (1 << np.arange(k - 1, -1, -1))).sum(axis=1)
    assert sorted(packed.tolist()) == list(range(order))


def test_qpsk_fixed_mapping():
    """QPSK is the most-tested edge of the table — pin exact bit→symbol values.

    Convention from constellations.py: bit_labels are MSB-first
    ``[q_bits, i_bits]``. For QPSK (side=2), Gray = [0, 1], so q_bit=i_bit=0
    selects the level −1, q_bit=i_bit=1 selects +1. Unit-power normalization
    divides by ``√2``.
    """
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    cases = {
        (0, 0): complex(-inv_sqrt2, -inv_sqrt2),  # q=0,i=0 → level (-1,-1)
        (0, 1): complex(+inv_sqrt2, -inv_sqrt2),
        (1, 0): complex(-inv_sqrt2, +inv_sqrt2),
        (1, 1): complex(+inv_sqrt2, +inv_sqrt2),
    }
    for bits_in, expected in cases.items():
        symbols = bits_to_symbols(np.array(bits_in, dtype=np.uint8), 4)
        np.testing.assert_allclose(symbols, [expected], atol=1e-12)


def test_16qam_corner_amplitudes():
    """The 4 corners of 16-QAM should sit at the maximum distance from origin."""
    constellation, _ = square_qam(16)
    amplitudes = np.sort(np.abs(constellation))
    # 16-QAM with normalization √(10) has corner amplitude √(9·2/10) = √(1.8).
    # 4 corners + 8 edges + 4 inner. The corners are the unique max.
    max_amp = amplitudes[-1]
    np.testing.assert_allclose(max_amp, np.sqrt(2.0 * 9.0 / 10.0), atol=1e-12)
    n_corners = int(np.sum(np.isclose(amplitudes, max_amp)))
    assert n_corners == 4


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_symbols_to_bits_inverse_of_bits_to_symbols(order: int):
    """For on-grid symbols, demap is the exact inverse of map."""
    rng = np.random.default_rng(0)
    k = bits_per_symbol(order)
    n = 200
    bits = rng.integers(0, 2, size=n * k, dtype=np.uint8)
    symbols = bits_to_symbols(bits, order)
    recovered = symbols_to_bits(symbols, order)
    np.testing.assert_array_equal(bits, recovered)


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_all_zero_bits_map_to_bottom_left_corner(order: int):
    """Gray code starts at 0 → level index 0 → minimum coordinate on each axis.

    So an all-zero bit pattern always maps to the bottom-left corner. This
    pins the bit ordering against accidental MSB/LSB swap regressions.
    Note: all-ones does NOT map to a corner for ``order ≥ 16`` because
    Gray code 0b11…1 inverts to binary 0b10…0 = side/2, an inner level.
    """
    k = bits_per_symbol(order)
    constellation, _ = square_qam(order)
    min_real = float(np.min(constellation.real))
    min_imag = float(np.min(constellation.imag))
    zeros = bits_to_symbols(np.zeros(k, dtype=np.uint8), order)
    np.testing.assert_allclose(zeros[0], complex(min_real, min_imag), atol=1e-12)


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_axis_nearest_neighbors_differ_by_one_bit(order: int):
    """For square-QAM with separable Gray coding, any two symbols whose
    constellation coordinates differ by one Δ on a single axis must differ
    in exactly one bit label. This is the *defining* Gray property and
    catches MSB/LSB swaps or row/column transposition bugs."""
    constellation, labels = square_qam(order)
    side = int(np.sqrt(order))
    # Δ between adjacent levels in the normalized constellation.
    levels = np.unique(np.round(constellation.real, 12))
    delta = float(np.diff(levels).min())

    def hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.sum(a != b))

    # Index-by-row reshape: in constellations.py sym_idx = q_idx * side + i_idx.
    grid_idx = np.arange(order).reshape(side, side)
    for q in range(side):
        for i in range(side - 1):
            assert hamming(labels[grid_idx[q, i]], labels[grid_idx[q, i + 1]]) == 1
            # Sanity-check the spacing assumption holds.
            np.testing.assert_allclose(
                constellation[grid_idx[q, i + 1]].real
                - constellation[grid_idx[q, i]].real,
                delta,
                atol=1e-12,
            )
    for i in range(side):
        for q in range(side - 1):
            assert hamming(labels[grid_idx[q, i]], labels[grid_idx[q + 1, i]]) == 1
