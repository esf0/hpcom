"""Pure-Python/NumPy implementations of hpcom's DSP hot-path kernels.

These are the reference kernels behind :class:`hpcom.backends.kernels.DSPKernels`.
A compiled sibling (e.g. ``hpcom.backends.rust``) implements the same three
functions with the same signatures and bit-exact outputs; the conformance
suite in ``tests/backends/test_kernel_conformance.py`` runs against every
registered kernel set.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Upper bound on the number of float64 distance entries evaluated at once in
#: :func:`nearest_indices` (≈ 32 MB of scratch memory).
_NEAREST_CHUNK_ELEMENTS = 1 << 22


def nearest_indices(
    received: npt.NDArray[np.complex128], points: npt.NDArray[np.complex128]
) -> npt.NDArray[np.intp]:
    """Index of the closest point (Euclidean) for every received sample.

    ``received`` is 1-D. Memory use is bounded by chunking over samples, so
    large constellations and long streams do not allocate an
    ``n_samples × M`` matrix. Ties resolve to the lowest index.
    """
    received = np.asarray(received, dtype=np.complex128)
    points = np.asarray(points, dtype=np.complex128)
    if received.ndim != 1 or points.ndim != 1:
        raise ValueError("nearest_indices expects 1-D received samples and 1-D points")
    out = np.empty(received.size, dtype=np.intp)
    if received.size == 0:
        return out
    pr, pi = points.real, points.imag
    step = max(1, _NEAREST_CHUNK_ELEMENTS // max(points.size, 1))
    for start in range(0, received.size, step):
        block = received[start : start + step]
        dist = (block.real[:, None] - pr) ** 2 + (block.imag[:, None] - pi) ** 2
        out[start : start + block.size] = np.argmin(dist, axis=1)
    return out


# ---------------------------------------------------------------------------
# Constant-composition distribution matching (multiset rank / unrank)
# ---------------------------------------------------------------------------
#
# For a block of n remaining positions with symbol counts c_j (Σ c_j = n), the
# number of distinct completions is the multinomial T = n! / Π c_j!. Fixing
# the next symbol to j leaves T·c_j/n completions (an exact integer), so in
# lexicographic order the completions that start with j occupy ranks
# [T·S_j/n, T·S_{j+1}/n) with S_j = Σ_{i<j} c_i. Hence the next symbol of the
# rank-r sequence is the j with S_j ≤ ⌊r·n/T⌋ < S_{j+1}. Each position costs
# a constant number of big-integer operations.


def _multinomial(composition: npt.NDArray[np.int64]) -> int:
    total = 1
    n = 0
    for c in (int(x) for x in composition):
        for k in range(1, c + 1):
            n += 1
            total = total * n // k
    return total


def _bits_to_int(bits: npt.NDArray[np.uint8]) -> int:
    """MSB-first bit array → Python int."""
    n = bits.size
    if n == 0:
        return 0
    pad = (-n) % 8
    packed = np.packbits(np.concatenate([np.zeros(pad, dtype=np.uint8), bits]))
    return int.from_bytes(packed.tobytes(), "big")


def _int_to_bits(value: int, n_bits: int) -> npt.NDArray[np.uint8]:
    """Python int → MSB-first bit array of length ``n_bits``."""
    n_bytes = (n_bits + 7) // 8
    raw = np.frombuffer(value.to_bytes(n_bytes, "big"), dtype=np.uint8)
    return np.unpackbits(raw)[-n_bits:] if n_bits else np.empty(0, dtype=np.uint8)


def ccdm_encode(
    bits: npt.NDArray[np.uint8],
    composition: npt.NDArray[np.int64],
    bits_per_block: int,
) -> npt.NDArray[np.intp]:
    """Map ``bits`` (a whole number of ``bits_per_block`` chunks) to symbol indices.

    Chunk ``b`` is read MSB-first as the rank of the output block among all
    permutations of the multiset ``composition``, in lexicographic order.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    composition = np.asarray(composition, dtype=np.int64)
    block_length = int(composition.sum())
    if bits_per_block <= 0 or bits.size % bits_per_block:
        raise ValueError(
            f"bits.size {bits.size} is not a multiple of bits_per_block {bits_per_block}"
        )
    total = _multinomial(composition)
    if bits_per_block > total.bit_length() - 1:
        raise ValueError("bits_per_block exceeds floor(log2(number of permutations))")
    n_blocks = bits.size // bits_per_block
    out = np.empty(n_blocks * block_length, dtype=np.intp)
    for b in range(n_blocks):
        rank = _bits_to_int(bits[b * bits_per_block : (b + 1) * bits_per_block])
        counts = composition.copy()
        t, n = total, block_length
        base = b * block_length
        for pos in range(block_length):
            q = rank * n // t
            cum = np.cumsum(counts)
            j = int(np.searchsorted(cum, q, side="right"))
            below = int(cum[j] - counts[j])  # S_j
            rank -= t * below // n
            t = t * int(counts[j]) // n
            counts[j] -= 1
            n -= 1
            out[base + pos] = j
    return out


def ccdm_decode(
    indices: npt.NDArray[np.intp],
    composition: npt.NDArray[np.int64],
    bits_per_block: int,
) -> npt.NDArray[np.uint8]:
    """Inverse of :func:`ccdm_encode`.

    Every block must be a permutation of ``composition``; a block with the
    wrong composition (e.g. after decision errors) raises ``ValueError`` —
    callers that must tolerate errors use
    :meth:`hpcom.modulation.ConstantCompositionDM.decode`, which handles
    invalid blocks explicitly.
    """
    indices = np.asarray(indices, dtype=np.intp)
    composition = np.asarray(composition, dtype=np.int64)
    block_length = int(composition.sum())
    if indices.size % block_length:
        raise ValueError(
            f"indices.size {indices.size} is not a multiple of block_length {block_length}"
        )
    total = _multinomial(composition)
    n_blocks = indices.size // block_length
    out = np.empty(n_blocks * bits_per_block, dtype=np.uint8)
    m = composition.size
    for b in range(n_blocks):
        block = indices[b * block_length : (b + 1) * block_length]
        if block.min(initial=0) < 0 or block.max(initial=0) >= m:
            raise ValueError(f"block {b} contains symbol indices outside [0, {m})")
        if not np.array_equal(np.bincount(block, minlength=m), composition):
            raise ValueError(f"block {b} does not have the DM composition")
        counts = composition.copy()
        t, n, rank = total, block_length, 0
        for sym in (int(s) for s in block):
            below = int(counts[:sym].sum())
            rank += t * below // n
            t = t * int(counts[sym]) // n
            counts[sym] -= 1
            n -= 1
        if rank >> bits_per_block:
            raise ValueError(f"block {b} has rank {rank} ≥ 2^{bits_per_block} (not a codeword)")
        out[b * bits_per_block : (b + 1) * bits_per_block] = _int_to_bits(rank, bits_per_block)
    return out
