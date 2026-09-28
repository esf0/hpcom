"""Distribution matchers for probabilistic shaping.

Provides a constant-composition distribution matcher (CC-DM): a bijective
encoder that maps uniform input bits to N-symbol blocks of a fixed composition
matching a target PMF. By construction, the *empirical* symbol distribution
equals the composition exactly, so the achieved PMF matches the target up to
the largest-remainder rounding error and is independent of how many blocks
are encoded.

This is the standard textbook predecessor to CCDM (Schulte & Böcherer,
2016). It uses combinatorial unrank/rank over multiset permutations rather
than arithmetic coding, which keeps the implementation small and obvious at
the cost of a few bits of rate (block-end overhead). Suitable for v1
shaping-gain experiments; a real CCDM would replace this transparently.

TODO(v2): vectorise _unrank_multiset/_rank_multiset with numpy; the current
Python loops are O(N × M) per block and will become a bottleneck for large
block lengths (N ≳ 256) or high-throughput simulations.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from hpcom._typing import ComplexArray

# ---------------------------------------------------------------------------
# Maxwell–Boltzmann PMF
# ---------------------------------------------------------------------------


def maxwell_boltzmann(constellation: ComplexArray, nu: float) -> npt.NDArray[np.float64]:
    """Maxwell–Boltzmann PMF over a constellation.

    For symbol energies ``E_i = |s_i|²`` the PMF is

    .. math:: p_i = \\frac{e^{-\\nu E_i}}{\\sum_j e^{-\\nu E_j}}

    This is the distribution that minimises ``E[E]`` (mean symbol energy)
    among all distributions on the constellation with the given entropy
    ``H = -Σ p_i log p_i`` — i.e., the AWGN-optimal shaping distribution.
    ``nu=0`` recovers the uniform distribution; larger ``nu`` concentrates
    probability on low-energy points.
    """
    if nu < 0:
        raise ValueError(f"nu must be >= 0, got {nu}")
    energies = np.abs(np.asarray(constellation, dtype=np.complex128)) ** 2
    if nu == 0.0:
        n = energies.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64)
    log_p = -nu * energies
    log_p -= log_p.max()  # numerical stability
    p = np.exp(log_p)
    p /= p.sum()
    return p


# ---------------------------------------------------------------------------
# Composition (target PMF → integer counts summing to N)
# ---------------------------------------------------------------------------


def _composition_from_pmf(pmf: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.int64]:
    """Largest-remainder rounding: integer counts ``n_i`` with ``Σn_i = N``
    and ``n_i ≈ N·p_i``. Returns a length-M int64 array."""
    raw = pmf * n
    floor = np.floor(raw).astype(np.int64)
    deficit = int(n - floor.sum())
    if deficit > 0:
        remainders = raw - floor
        top = np.argsort(-remainders, kind="stable")[:deficit]
        floor[top] += 1
    return floor


# ---------------------------------------------------------------------------
# Multiset rank / unrank
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _multinomial_with_counts(total: int, counts: tuple[int, ...]) -> int:
    """``total! / Π(c_i!)`` with cache. ``sum(counts) == total`` is required."""
    if sum(counts) != total:
        raise ValueError(f"counts sum {sum(counts)} != total {total}")
    num = math.factorial(total)
    den = 1
    for c in counts:
        if c > 0:
            den *= math.factorial(c)
    return num // den


def _unrank_multiset(rank: int, composition: list[int]) -> list[int]:
    """Return the ``rank``-th multiset permutation (0-indexed, lex order).

    ``composition`` is mutated/copied internally; the caller should pass a
    fresh list. The returned list has length ``sum(composition_in)``.
    """
    comp = list(composition)
    n_left = sum(comp)
    total = _multinomial_with_counts(n_left, tuple(comp))
    if not (0 <= rank < total):
        raise ValueError(f"rank {rank} is out of range [0, {total})")
    result: list[int] = []
    while n_left > 0:
        for j in range(len(comp)):
            if comp[j] == 0:
                continue
            comp[j] -= 1
            count = _multinomial_with_counts(n_left - 1, tuple(comp))
            if rank < count:
                result.append(j)
                n_left -= 1
                break
            rank -= count
            comp[j] += 1
    return result


def _rank_multiset(perm: list[int], composition: list[int]) -> int:
    """Inverse of :func:`_unrank_multiset`."""
    m = len(composition)
    for sym in perm:
        if sym < 0 or sym >= m:
            raise ValueError(f"symbol index {sym} is out of range [0, {m})")
    comp = list(composition)
    n_left = sum(comp)
    rank = 0
    for sym in perm:
        for j in range(sym):
            if comp[j] == 0:
                continue
            comp[j] -= 1
            rank += _multinomial_with_counts(n_left - 1, tuple(comp))
            comp[j] += 1
        comp[sym] -= 1
        n_left -= 1
    return rank


# ---------------------------------------------------------------------------
# Bit ↔ int packing (MSB-first)
# ---------------------------------------------------------------------------


def _bits_to_int(bits: npt.NDArray[np.uint8]) -> int:
    """MSB-first bit array → Python int."""
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _int_to_bits(value: int, k: int) -> npt.NDArray[np.uint8]:
    """Python int → MSB-first length-k bit array."""
    out = np.empty(k, dtype=np.uint8)
    for i in range(k - 1, -1, -1):
        out[i] = value & 1
        value >>= 1
    if value != 0:
        raise ValueError(f"value does not fit in {k} bits")
    return out


# ---------------------------------------------------------------------------
# CC-DM
# ---------------------------------------------------------------------------


class ConstantCompositionDM(BaseModel):
    """Bijective bit↔symbol-index encoder with a fixed per-block composition.

    Constructed from a target PMF and a block length ``N``. The composition
    ``(n_0, …, n_{M-1})`` with ``Σ n_i = N`` is chosen by largest-remainder
    rounding so that ``n_i ≈ N·p_i``. Each N-symbol block is one of the
    ``C = N! / Π n_i!`` multiset permutations; ``bits_per_block =
    ⌊log₂ C⌋`` bits encode the rank.

    Encoding: split the input bit stream into ``bits_per_block``-sized chunks,
    interpret each as an integer rank, unrank to a multiset permutation.
    Decoding rank↔permutation is the inverse.

    Empirical PMF over any whole number of blocks equals
    ``composition / N`` exactly, independent of the input bits.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    target_pmf: npt.NDArray[np.float64]
    block_length: int
    # Derived fields, populated in model_validator. Defaults are sentinels
    # only — they are always overwritten before the model is returned.
    composition: npt.NDArray[np.int64] = None  # type: ignore[assignment]
    bits_per_block: int = 0

    @field_validator("target_pmf", mode="before")
    @classmethod
    def _coerce_pmf(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)

    @model_validator(mode="after")
    def _validate_and_compute(self) -> ConstantCompositionDM:
        pmf = self.target_pmf
        if pmf.ndim != 1 or pmf.size < 2:
            raise ValueError(f"target_pmf must be 1-D with ≥2 entries, got shape {pmf.shape}")
        if not np.isclose(pmf.sum(), 1.0, atol=1e-12):
            raise ValueError(f"target_pmf must sum to 1, got {pmf.sum():.6e}")
        if (pmf < 0).any():
            raise ValueError("target_pmf entries must be non-negative")
        if self.block_length < pmf.size:
            raise ValueError(
                f"block_length ({self.block_length}) must be ≥ number of symbols "
                f"({pmf.size}) to support a non-trivial composition"
            )

        composition = _composition_from_pmf(pmf, self.block_length)
        total = _multinomial_with_counts(self.block_length, tuple(int(c) for c in composition))
        if total < 2:
            raise ValueError(
                "block has a single permutation (composition is degenerate); "
                "increase block_length or use a less peaked PMF"
            )
        # Assign derived state through __dict__ — model_validator runs before
        # pydantic enforces frozen on setattr.
        self.__dict__["composition"] = composition
        self.__dict__["bits_per_block"] = math.floor(math.log2(total))
        return self

    @property
    def _composition_list(self) -> list[int]:
        """Python-list copy of composition (cached at first access)."""
        # Memoise via __dict__ since the model is frozen.
        cached = self.__dict__.get("_composition_list_cache")
        if cached is None:
            cached = self.composition.tolist()
            self.__dict__["_composition_list_cache"] = cached
        return cached  # type: ignore[no-any-return]

    @property
    def empirical_pmf(self) -> npt.NDArray[np.float64]:
        """``composition / block_length`` — the exact distribution emitted."""
        return self.composition.astype(np.float64) / self.block_length

    @property
    def rate_bits_per_symbol(self) -> float:
        """Information rate the encoder achieves (bits per emitted symbol)."""
        return self.bits_per_block / self.block_length

    def encode(self, bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.intp]:
        """Encode a bit stream → symbol-index stream.

        ``bits.size`` must be a multiple of :attr:`bits_per_block`. Output
        length is ``(bits.size / bits_per_block) * block_length``.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if bits.size % self.bits_per_block != 0:
            raise ValueError(
                f"bits.size {bits.size} is not a multiple of bits_per_block {self.bits_per_block}"
            )
        n_blocks = bits.size // self.bits_per_block
        out = np.empty(n_blocks * self.block_length, dtype=np.intp)
        for i in range(n_blocks):
            chunk = bits[i * self.bits_per_block : (i + 1) * self.bits_per_block]
            rank = _bits_to_int(chunk)
            perm = _unrank_multiset(rank, self._composition_list)
            out[i * self.block_length : (i + 1) * self.block_length] = perm
        return out

    def decode(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.uint8]:
        """Inverse of :meth:`encode`."""
        indices = np.asarray(indices, dtype=np.intp)
        if indices.size % self.block_length != 0:
            raise ValueError(
                f"indices.size {indices.size} is not a multiple of block_length {self.block_length}"
            )
        n_blocks = indices.size // self.block_length
        out = np.empty(n_blocks * self.bits_per_block, dtype=np.uint8)
        for i in range(n_blocks):
            perm = indices[i * self.block_length : (i + 1) * self.block_length].tolist()
            rank = _rank_multiset(perm, self._composition_list)
            out[i * self.bits_per_block : (i + 1) * self.bits_per_block] = _int_to_bits(
                rank, self.bits_per_block
            )
        return out


__all__ = ["ConstantCompositionDM", "maxwell_boltzmann"]
