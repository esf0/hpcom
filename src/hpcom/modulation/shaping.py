"""Constellations: points, bit labels, and (optional) symbol probabilities.

:class:`ConstellationConfig` is the one object the transmitter and receiver
use to map between bits, symbol indices, and complex points. Square QAM is
built with :meth:`ConstellationConfig.from_square_qam`; geometric shaping
through arbitrary points (:meth:`~ConstellationConfig.from_points`,
:func:`apsk`); probabilistic shaping through ``probabilities`` (see
:mod:`hpcom.modulation.dm`).
"""

from __future__ import annotations

from functools import cache, cached_property

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from hpcom._model import ComplexArrayField, FloatArrayField, HpcomModel, UInt8ArrayField
from hpcom._typing import ComplexArray, _is_power_of_two
from hpcom.modulation.constellations import DEFAULT_QAM_LABELING, SquareQAMLabeling, square_qam
from hpcom.modulation.mapping import nearest_symbol_indices, pack_bits_msb_first

#: Tolerance on ``E[|s|²] = 1`` and on ``Σ p = 1``.
_UNIT_TOL = 1e-9


def _expected_power(points: ComplexArray, probabilities: npt.NDArray[np.float64] | None) -> float:
    """``E[|s|²]`` under ``probabilities`` (uniform if ``None``)."""
    energy = np.abs(points) ** 2
    if probabilities is None:
        return float(energy.mean())
    return float(np.dot(probabilities, energy))


class ConstellationConfig(HpcomModel):
    """Validated constellation.

    Invariants (checked at construction):

    * ``points``: shape ``(M,)``, ``M`` a power of two, all points distinct.
    * ``bit_labels``: shape ``(M, log₂M)``, entries 0/1, rows unique; row
      ``i`` labels ``points[i]`` (MSB first).
    * ``probabilities`` (optional): shape ``(M,)``, non-negative, sums to 1.
    * ``E[|s|²] = 1`` under ``probabilities`` (uniform when ``None``). The
      ``from_*`` factories normalise for you.

    With ``probabilities`` set, a transmitter draws symbols from that PMF —
    i.i.d. unless a distribution matcher (``dm``) is configured.
    """

    points: ComplexArrayField
    bit_labels: UInt8ArrayField
    probabilities: FloatArrayField | None = None

    @model_validator(mode="after")
    def _validate(self) -> ConstellationConfig:
        m = self.points.shape[0] if self.points.ndim == 1 else -1
        if self.points.ndim != 1 or not _is_power_of_two(m):
            raise ValueError(
                f"points must be 1-D with a power-of-2 length, got shape {self.points.shape}"
            )
        if np.unique(self.points).size != m:
            raise ValueError("constellation points must be distinct")
        k = m.bit_length() - 1
        if self.bit_labels.shape != (m, k):
            raise ValueError(f"bit_labels must have shape ({m}, {k}), got {self.bit_labels.shape}")
        if np.any(self.bit_labels > 1):
            raise ValueError("bit_labels must contain only 0/1 entries")
        if np.unique(pack_bits_msb_first(self.bit_labels)).size != m:
            raise ValueError("bit_labels rows must be unique (one label per symbol)")

        p = self.probabilities
        if p is not None:
            if p.shape != (m,):
                raise ValueError(f"probabilities must have shape ({m},), got {p.shape}")
            if np.any(p < 0) or not np.all(np.isfinite(p)):
                raise ValueError("probabilities must be finite and non-negative")
            if abs(float(p.sum()) - 1.0) > _UNIT_TOL:
                raise ValueError(f"probabilities must sum to 1, got {p.sum():.12g}")

        power = _expected_power(self.points, p)
        if abs(power - 1.0) > _UNIT_TOL:
            raise ValueError(
                f"expected symbol power E[|s|²] must be 1 (got {power:.12g}); "
                "use ConstellationConfig.from_points to normalise automatically"
            )
        return self

    # ---- properties ------------------------------------------------------

    @property
    def modulation_order(self) -> int:
        """Number of points ``M``."""
        return int(self.points.shape[0])

    @property
    def bits_per_symbol(self) -> int:
        """``log₂ M``."""
        return int(self.bit_labels.shape[1])

    @property
    def is_uniform(self) -> bool:
        """``True`` when no ``probabilities`` are set (equiprobable points)."""
        return self.probabilities is None

    @property
    def pmf(self) -> npt.NDArray[np.float64]:
        """Symbol probabilities (uniform when ``probabilities`` is ``None``)."""
        if self.probabilities is None:
            return np.full(self.modulation_order, 1.0 / self.modulation_order)
        return self.probabilities

    @property
    def entropy_bits(self) -> float:
        """Source entropy ``H(X) = −Σ p log₂ p`` in bits/symbol."""
        p = self.pmf[self.pmf > 0]
        return float(-np.sum(p * np.log2(p)))

    @cached_property
    def _index_of_label(self) -> npt.NDArray[np.intp]:
        """Packed label value → point index."""
        lut = np.empty(self.modulation_order, dtype=np.intp)
        lut[pack_bits_msb_first(self.bit_labels)] = np.arange(self.modulation_order)
        return lut

    # ---- factories -------------------------------------------------------

    @classmethod
    def from_square_qam(
        cls, modulation_order: int, labeling: SquareQAMLabeling = DEFAULT_QAM_LABELING
    ) -> ConstellationConfig:
        """Gray-coded square QAM (shared, cached instance).

        ``labeling`` selects the bit-labelling convention
        (:data:`~hpcom.modulation.constellations.SquareQAMLabeling`); the
        default reproduces pre-v1 hpcom. Transmitters and receivers built with
        ``modulation_order=M`` use the default; pass
        ``constellation=ConstellationConfig.from_square_qam(M, labeling=...)``
        to select the other.
        """
        return _square_qam_config(modulation_order, labeling)

    @classmethod
    def from_points(
        cls,
        points: npt.ArrayLike,
        bit_labels: npt.ArrayLike | None = None,
        probabilities: npt.ArrayLike | None = None,
    ) -> ConstellationConfig:
        """Build a constellation from arbitrary points (geometric shaping).

        ``points`` are rescaled to ``E[|s|²] = 1`` under ``probabilities``.
        Without ``bit_labels`` the natural binary labelling (row ``i`` =
        binary of ``i``) is used — valid, but not Gray; pass labels for good
        BER.
        """
        pts = np.asarray(points, dtype=np.complex128)
        prob = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)
        if pts.ndim != 1 or not _is_power_of_two(pts.shape[0]):
            raise ValueError(f"points must be 1-D with a power-of-2 length, got shape {pts.shape}")
        if prob is not None and prob.shape != pts.shape:
            raise ValueError(f"probabilities must have shape {pts.shape}, got {prob.shape}")
        power = _expected_power(pts, prob)
        if not power > 0:
            raise ValueError("constellation has zero expected power")
        labels = (
            natural_labels(pts.shape[0]) if bit_labels is None else np.asarray(bit_labels, np.uint8)
        )
        return cls(points=pts / np.sqrt(power), bit_labels=labels, probabilities=prob)

    def with_probabilities(self, probabilities: npt.ArrayLike | None) -> ConstellationConfig:
        """Same geometry and labels with a new PMF (points re-normalised to unit power)."""
        return ConstellationConfig.from_points(self.points, self.bit_labels, probabilities)

    # ---- mapping ---------------------------------------------------------

    def indices_from_bits(self, bits: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Split ``bits`` (flat, length multiple of ``log₂M``) into labels → point indices."""
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        k = self.bits_per_symbol
        if bits.size % k:
            raise ValueError(f"number of bits ({bits.size}) is not a multiple of {k}")
        if np.any(bits > 1):
            raise ValueError("bits must be 0/1")
        return self._index_of_label[pack_bits_msb_first(bits.reshape(-1, k))]

    def bits_from_indices(self, indices: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        """Concatenated labels of ``indices`` (flat, MSB first)."""
        return self.bit_labels[np.asarray(indices, dtype=np.intp)].reshape(-1)

    def map(self, bits: npt.ArrayLike) -> ComplexArray:
        """Bits → symbols."""
        return self.points[self.indices_from_bits(bits)]

    def nearest(self, received: ComplexArray) -> npt.NDArray[np.intp]:
        """Minimum-Euclidean-distance decisions (point indices), any input shape."""
        return nearest_symbol_indices(received, self.points)

    def demap(self, received: ComplexArray) -> npt.NDArray[np.uint8]:
        """Hard-decision demap to bits (flat). Uses minimum distance, not MAP."""
        return self.bits_from_indices(self.nearest(np.asarray(received).reshape(-1)))


@cache
def _square_qam_config(modulation_order: int, labeling: SquareQAMLabeling) -> ConstellationConfig:
    points, labels = square_qam(modulation_order, labeling)
    return ConstellationConfig(points=points, bit_labels=labels)


def natural_labels(m: int) -> npt.NDArray[np.uint8]:
    """Natural binary labels: row ``i`` is the ``log₂m``-bit binary of ``i`` (MSB first)."""
    k = m.bit_length() - 1
    return ((np.arange(m)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)


def apsk(
    ring_sizes: tuple[int, ...] | list[int],
    ring_radii: tuple[float, ...] | list[float],
    ring_phases: tuple[float, ...] | list[float] | None = None,
) -> ConstellationConfig:
    """Amplitude-phase shift keying: concentric rings of equally spaced points.

    Ring ``i`` carries ``ring_sizes[i]`` points at radius ``ring_radii[i]``
    and angles ``ring_phases[i] + 2πk/ring_sizes[i]``. ``sum(ring_sizes)``
    must be a power of two. Points are ordered ring by ring.

    .. warning::
       Labels are natural binary, **not** Gray; pass explicit labels to
       :meth:`ConstellationConfig.from_points` when BER matters.

    >>> apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85)).modulation_order
    16
    """
    if len(ring_sizes) != len(ring_radii):
        raise ValueError(
            f"ring_sizes and ring_radii must have the same length, "
            f"got {len(ring_sizes)} and {len(ring_radii)}"
        )
    phases = [0.0] * len(ring_sizes) if ring_phases is None else list(ring_phases)
    if len(phases) != len(ring_sizes):
        raise ValueError(f"ring_phases must have length {len(ring_sizes)}, got {len(phases)}")
    if any(n < 1 for n in ring_sizes):
        raise ValueError(f"all ring_sizes must be >= 1, got {ring_sizes}")
    if any(r <= 0 for r in ring_radii):
        raise ValueError(f"all ring_radii must be > 0, got {ring_radii}")
    total = int(sum(ring_sizes))
    if not _is_power_of_two(total):
        raise ValueError(f"sum(ring_sizes) must be a power of 2, got {total}")
    rings = [
        r * np.exp(1j * (ph + 2.0 * np.pi * np.arange(n) / n))
        for n, r, ph in zip(ring_sizes, ring_radii, phases, strict=True)
    ]
    return ConstellationConfig.from_points(np.concatenate(rings))


__all__ = ["ConstellationConfig", "apsk", "natural_labels"]
