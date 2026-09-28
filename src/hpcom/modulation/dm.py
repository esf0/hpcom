"""Probabilistic shaping: Maxwell–Boltzmann PMFs and constant-composition matching.

:class:`ConstantCompositionDM` is a bijective map from uniform bits to blocks
of ``N`` symbol indices that all have the same composition — ``n_i``
occurrences of symbol ``i`` with ``n_i ≈ N·p_i`` (largest-remainder rounding
of the target PMF). Block ``b`` encodes ``bits_per_block = ⌊log₂ C⌋`` bits
as its lexicographic rank among the ``C = N!/Π n_i!`` permutations of the
multiset. The emitted distribution is therefore exactly ``n_i/N`` over any
whole number of blocks, independent of the data.

This is the enumerative (rank/unrank) form of constant-composition
distribution matching (CCDM; Schulte & Böcherer, *IEEE Trans. Inf. Theory*
62(1), 2016, use arithmetic coding for the same map). The rate loss relative
to the entropy vanishes as ``N`` grows (≈ ``(M/2)·log₂N / N``).
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy.optimize import brentq

from hpcom._model import FloatArrayField, HpcomModel
from hpcom._typing import ComplexArray
from hpcom.backends.kernels import get_kernels


def maxwell_boltzmann(points: npt.ArrayLike, nu: float) -> npt.NDArray[np.float64]:
    r"""Maxwell–Boltzmann PMF :math:`p_i \propto e^{-\nu |s_i|^2}` over ``points``.

    For a given entropy this is the distribution that minimises the mean
    symbol energy — the AWGN-optimal shaping distribution. ``nu = 0`` is
    uniform; larger ``nu`` concentrates probability on low-energy points.
    """
    if nu < 0:
        raise ValueError(f"nu must be >= 0, got {nu}")
    energy = np.abs(np.asarray(points, dtype=np.complex128)) ** 2
    log_p = -nu * energy
    log_p -= log_p.max()
    p = np.exp(log_p)
    out: npt.NDArray[np.float64] = p / p.sum()
    return out


def maxwell_boltzmann_for_entropy(
    points: npt.ArrayLike, entropy_bits: float
) -> npt.NDArray[np.float64]:
    """Maxwell–Boltzmann PMF over ``points`` with the requested entropy (bits/symbol).

    ``entropy_bits`` must lie in ``(0, log₂M]``; ``log₂M`` returns the
    uniform PMF. The rate parameter ν is found by root bracketing (the
    entropy is monotonically decreasing in ν).
    """
    pts = np.asarray(points, dtype=np.complex128)
    h_max = float(np.log2(pts.size))
    if not 0 < entropy_bits <= h_max + 1e-12:
        raise ValueError(f"entropy_bits must be in (0, {h_max}], got {entropy_bits}")
    if entropy_bits >= h_max - 1e-12:
        return maxwell_boltzmann(pts, 0.0)

    def excess(nu: float) -> float:
        p = maxwell_boltzmann(pts, nu)
        nz = p[p > 0]
        return float(-np.sum(nz * np.log2(nz))) - entropy_bits

    hi = 1.0
    while excess(hi) > 0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(f"entropy {entropy_bits} is not reachable for this constellation")
    return maxwell_boltzmann(pts, brentq(excess, 0.0, hi, xtol=1e-14, rtol=1e-14))


def composition_from_pmf(pmf: npt.ArrayLike, block_length: int) -> npt.NDArray[np.int64]:
    """Integer counts ``n_i`` with ``Σ n_i = N`` and ``n_i ≈ N·p_i`` (largest remainder)."""
    pmf = np.asarray(pmf, dtype=np.float64)
    raw = pmf * block_length
    counts: npt.NDArray[np.int64] = np.floor(raw).astype(np.int64)
    deficit = int(block_length - counts.sum())
    if deficit > 0:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:deficit]] += 1
    return counts


class ConstantCompositionDM(HpcomModel):
    """Constant-composition distribution matcher (see module docstring).

    Attributes
    ----------
    target_pmf
        Desired symbol PMF (length ``M``, sums to 1). Must match the
        ``probabilities`` of the constellation it is used with.
    block_length
        Symbols per block ``N``.
    """

    target_pmf: FloatArrayField
    block_length: int = Field(ge=2)

    @model_validator(mode="after")
    def _validate(self) -> ConstantCompositionDM:
        p = self.target_pmf
        if p.ndim != 1 or p.size < 2:
            raise ValueError(f"target_pmf must be 1-D with >= 2 entries, got shape {p.shape}")
        if np.any(p < 0) or not np.all(np.isfinite(p)):
            raise ValueError("target_pmf entries must be finite and non-negative")
        if abs(float(p.sum()) - 1.0) > 1e-9:
            raise ValueError(f"target_pmf must sum to 1, got {p.sum():.12g}")
        n_support = int(np.count_nonzero(p))
        if self.block_length < n_support:
            raise ValueError(
                f"block_length ({self.block_length}) must be >= the number of symbols with "
                f"non-zero target probability ({n_support}); shorter blocks cannot contain them all"
            )
        if self.bits_per_block < 1:
            raise ValueError(
                f"composition {self.composition.tolist()} admits a single permutation "
                "(zero rate); increase block_length or use a less peaked PMF"
            )
        return self

    # ---- derived quantities --------------------------------------------

    @cached_property
    def composition(self) -> npt.NDArray[np.int64]:
        """Symbol counts ``n_i`` per block (read-only)."""
        counts = composition_from_pmf(self.target_pmf, self.block_length)
        counts.flags.writeable = False
        return counts

    @cached_property
    def n_permutations(self) -> int:
        """Number of distinct blocks ``C = N! / Π n_i!`` (exact integer)."""
        total = 1
        n = 0
        for c in self.composition.tolist():
            for k in range(1, c + 1):
                n += 1
                total = total * n // k
        return total

    @property
    def bits_per_block(self) -> int:
        """Information bits per block, ``⌊log₂ C⌋``."""
        return self.n_permutations.bit_length() - 1

    @property
    def empirical_pmf(self) -> npt.NDArray[np.float64]:
        """Distribution actually emitted, ``n_i / N``."""
        return self.composition.astype(np.float64) / self.block_length

    @property
    def rate_bits_per_symbol(self) -> float:
        """Information rate ``bits_per_block / N`` in bits per symbol."""
        return self.bits_per_block / self.block_length

    @property
    def rate_loss_bits_per_symbol(self) -> float:
        """``H(empirical_pmf) − rate``: the matcher's rate loss."""
        p = self.empirical_pmf[self.empirical_pmf > 0]
        return float(-np.sum(p * np.log2(p))) - self.rate_bits_per_symbol

    # ---- encode / decode ------------------------------------------------

    def encode(self, bits: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Bits (a multiple of :attr:`bits_per_block`) → symbol indices."""
        return get_kernels().ccdm_encode(
            np.asarray(bits, dtype=np.uint8).reshape(-1), self.composition, self.bits_per_block
        )

    def decode(self, indices: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        """Symbol indices (a multiple of ``block_length``) → bits.

        Raises ``ValueError`` if any block is not a codeword (wrong
        composition, or rank ≥ 2^bits_per_block) — e.g. after decision
        errors. Use :meth:`decode_with_errors` on noisy decisions.
        """
        return get_kernels().ccdm_decode(
            np.asarray(indices, dtype=np.intp).reshape(-1), self.composition, self.bits_per_block
        )

    def decode_with_errors(
        self, indices: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        """Decode noisy decisions block by block.

        Returns ``(bits, valid)``: blocks that are not codewords cannot be
        inverted — their bits are set to 0 and ``valid[b]`` is ``False``.
        (A real system runs FEC before the DM decoder; pre-FEC, a corrupted
        block is simply lost.)
        """
        idx = np.asarray(indices, dtype=np.intp).reshape(-1)
        n = self.block_length
        if idx.size % n:
            raise ValueError(f"indices.size {idx.size} is not a multiple of block_length {n}")
        blocks = idx.reshape(-1, n)
        m = self.composition.size
        valid = np.array(
            [
                b.min() >= 0
                and b.max() < m
                and np.array_equal(np.bincount(b, minlength=m), self.composition)
                for b in blocks
            ],
            dtype=bool,
        )
        k = self.bits_per_block
        bits = np.zeros((blocks.shape[0], k), dtype=np.uint8)
        kernels = get_kernels()
        for i in np.flatnonzero(valid):
            try:
                bits[i] = kernels.ccdm_decode(blocks[i], self.composition, k)
            except ValueError:  # right composition but rank ≥ 2^k: not a codeword
                valid[i] = False
        return bits.reshape(-1), valid


def empirical_symbol_energy(points: ComplexArray, pmf: npt.ArrayLike) -> float:
    """``Σ p_i |s_i|²``."""
    return float(np.dot(np.asarray(pmf, dtype=np.float64), np.abs(points) ** 2))


__all__ = [
    "ConstantCompositionDM",
    "composition_from_pmf",
    "maxwell_boltzmann",
    "maxwell_boltzmann_for_entropy",
]
