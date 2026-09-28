"""Conformance suite for DSP kernel sets (``hpcom.backends.kernels``).

Every registered kernel set must reproduce brute-force references exactly
(integer outputs, so equality — not tolerances). A compiled kernel set added
later is picked up automatically.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from hpcom.backends import kernels as kernel_registry
from hpcom.backends.kernels import get_kernels, list_kernels, register_kernels, set_default_kernels
from hpcom.modulation import ConstellationConfig, apsk


@pytest.fixture(params=list_kernels())
def k(request):
    return get_kernels(request.param)


def _int_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


# ---- nearest_indices --------------------------------------------------------


@pytest.mark.parametrize(
    "points",
    [
        ConstellationConfig.from_square_qam(4).points,
        ConstellationConfig.from_square_qam(64).points,
        apsk((4, 12), (1.0, 2.85)).points,
    ],
)
def test_nearest_indices_matches_brute_force(k, points):
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(5000) + 1j * rng.standard_normal(5000)) * 0.8
    expected = np.argmin(np.abs(y[:, None] - points[None, :]), axis=1)
    np.testing.assert_array_equal(k.nearest_indices(y, points), expected)


def test_nearest_indices_bounded_memory_on_long_streams(k, monkeypatch):
    """Chunking must give identical results across chunk boundaries."""
    import hpcom.backends.python.kernels as pk

    monkeypatch.setattr(pk, "_NEAREST_CHUNK_ELEMENTS", 64 * 7)  # 7 samples per chunk
    pts = ConstellationConfig.from_square_qam(64).points
    y = np.random.default_rng(1).standard_normal(1001) + 0j
    np.testing.assert_array_equal(
        k.nearest_indices(y, pts), np.argmin(np.abs(y[:, None] - pts), axis=1)
    )


def test_nearest_indices_empty_input(k):
    assert k.nearest_indices(np.empty(0, complex), np.array([1 + 0j, -1 + 0j])).size == 0


def test_nearest_indices_exact_points_map_to_themselves(k):
    pts = ConstellationConfig.from_square_qam(256).points
    np.testing.assert_array_equal(k.nearest_indices(pts, pts), np.arange(256))


# ---- CC-DM rank / unrank -----------------------------------------------------


@pytest.mark.parametrize("composition", [(2, 1, 1), (3, 2), (1, 1, 1, 1), (4, 0, 2, 1)])
def test_ccdm_unrank_enumerates_permutations_in_lexicographic_order(k, composition):
    comp = np.array(composition, dtype=np.int64)
    symbols = [s for s, c in enumerate(composition) for _ in range(c)]
    perms = sorted(set(itertools.permutations(symbols)))
    assert len(perms) == math.factorial(sum(composition)) // math.prod(
        math.factorial(c) for c in composition
    )
    width = len(perms).bit_length() - 1
    for rank in range(2**width):
        out = k.ccdm_encode(_int_bits(rank, width), comp, width)
        assert tuple(out) == perms[rank]
        np.testing.assert_array_equal(k.ccdm_decode(out, comp, width), _int_bits(rank, width))


@pytest.mark.parametrize(("m", "n"), [(4, 64), (16, 256), (64, 512)])
def test_ccdm_round_trip_many_blocks(k, m, n):
    rng = np.random.default_rng(m)
    comp = rng.multinomial(n, np.full(m, 1.0 / m)).astype(np.int64)
    total = math.factorial(n) // math.prod(math.factorial(int(c)) for c in comp)
    width = total.bit_length() - 1
    bits = rng.integers(0, 2, width * 7, dtype=np.uint8)
    idx = k.ccdm_encode(bits, comp, width)
    for b in range(7):
        np.testing.assert_array_equal(np.bincount(idx[b * n : (b + 1) * n], minlength=m), comp)
    np.testing.assert_array_equal(k.ccdm_decode(idx, comp, width), bits)


def test_ccdm_encode_rejects_partial_block(k):
    with pytest.raises(ValueError, match="multiple of bits_per_block"):
        k.ccdm_encode(np.zeros(5, np.uint8), np.array([2, 2]), 2)


def test_ccdm_encode_rejects_too_many_bits_per_block(k):
    with pytest.raises(ValueError, match="exceeds"):
        k.ccdm_encode(np.zeros(3, np.uint8), np.array([2, 2]), 3)  # C = 6 → at most 2 bits


def test_ccdm_decode_rejects_wrong_composition(k):
    with pytest.raises(ValueError, match="composition"):
        k.ccdm_decode(np.array([0, 0, 0, 1]), np.array([2, 2]), 2)


def test_ccdm_decode_rejects_rank_beyond_codebook(k):
    comp = np.array([2, 2])  # 6 permutations, 2 bits → ranks 4, 5 are not codewords
    last = np.array([1, 1, 0, 0])  # rank 5
    with pytest.raises(ValueError, match="not a codeword"):
        k.ccdm_decode(last, comp, 2)


# ---- registry ----------------------------------------------------------------


def test_python_kernels_are_the_default():
    assert get_kernels() is get_kernels("python")


def test_register_rejects_incomplete_kernel_set():
    with pytest.raises(TypeError, match="missing"):
        register_kernels("broken", object())


def test_unknown_kernel_set():
    with pytest.raises(KeyError, match="unknown kernel set"):
        get_kernels("nope")


def test_switching_default_kernel_set(monkeypatch):
    monkeypatch.setattr(kernel_registry, "_registry", dict(kernel_registry._registry))
    monkeypatch.setattr(kernel_registry, "_default", "python")
    py = get_kernels("python")
    register_kernels("alias", py)
    set_default_kernels("alias")
    assert get_kernels() is py
    with pytest.raises(KeyError):
        set_default_kernels("nope")
