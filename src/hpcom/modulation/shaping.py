"""Constellation-shaping primitives.

Provides :class:`ConstellationConfig` — a validated, immutable container for
``(points, bit_labels, probabilities)`` that the rest of the modulation API
can consume in place of a bare ``modulation_order``. Geometric shaping
plugs in through arbitrary ``points``; probabilistic shaping through a
non-uniform ``probabilities`` vector.

This module is pure NumPy; framework-specific code lives in
``hpcom.backends``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from hpcom._typing import ComplexArray, _is_power_of_two
from hpcom.modulation.constellations import bits_per_symbol, square_qam
from hpcom.modulation.mapping import _pack_bit_groups_msb_first, nearest_symbol_indices

_POWER_TOL = 1e-9


def _expected_power(
    points: ComplexArray, probabilities: npt.NDArray[np.float64] | None
) -> float:
    """E[|s|²] under the symbol distribution (uniform if probabilities is None)."""
    sq = np.abs(points) ** 2
    if probabilities is None:
        return float(sq.mean())
    return float(np.dot(probabilities, sq))


class ConstellationConfig(BaseModel):
    """Validated constellation: points, bit→symbol labels, optional priors.

    Invariants enforced at construction:

    * ``points`` has shape ``(M,)``, complex, with ``M`` a power of 2.
    * ``bit_labels`` has shape ``(M, log₂ M)`` ``uint8`` with all-unique rows.
    * ``probabilities`` (if given) has shape ``(M,)``, is non-negative, and
      sums to 1.
    * Expected symbol power ``E[|s|²]`` equals 1 within ``1e-9``. (Use the
      ``from_*`` factories; they normalise.)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    points: ComplexArray
    bit_labels: npt.NDArray[np.uint8]
    probabilities: npt.NDArray[np.float64] | None = None

    _SUPPORTED_QAM_ORDERS: ClassVar[tuple[int, ...]] = (4, 16, 64, 256, 1024)

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=np.complex128)

    @field_validator("bit_labels", mode="before")
    @classmethod
    def _coerce_bit_labels(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=np.uint8)

    @field_validator("probabilities", mode="before")
    @classmethod
    def _coerce_probabilities(cls, v: object) -> np.ndarray | None:
        if v is None:
            return None
        return np.asarray(v, dtype=np.float64)

    @model_validator(mode="after")
    def _validate(self) -> ConstellationConfig:
        points = self.points
        bit_labels = self.bit_labels

        m = points.shape[0]
        if points.ndim != 1 or not _is_power_of_two(m):
            raise ValueError(
                f"points must be 1-D with power-of-2 length, got shape {points.shape}"
            )

        k = int(np.log2(m))
        if bit_labels.shape != (m, k):
            raise ValueError(
                f"bit_labels must have shape ({m}, {k}), got {bit_labels.shape}"
            )
        if ((bit_labels != 0) & (bit_labels != 1)).any():
            raise ValueError("bit_labels must contain only 0/1 entries")

        packed = _pack_bit_groups_msb_first(bit_labels)
        if np.unique(packed).size != m:
            raise ValueError("bit_labels rows must be unique (one label per symbol)")

        probabilities = self.probabilities
        if probabilities is not None:
            if probabilities.shape != (m,):
                raise ValueError(
                    f"probabilities must have shape ({m},), got {probabilities.shape}"
                )
            if (probabilities < 0).any():
                raise ValueError("probabilities must be non-negative")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
                raise ValueError(
                    f"probabilities must sum to 1, got {probabilities.sum():.6e}"
                )

        power = _expected_power(points, probabilities)
        if not np.isclose(power, 1.0, atol=_POWER_TOL):
            raise ValueError(
                f"expected symbol power must be 1.0 (got {power:.6e}); use "
                "ConstellationConfig.from_points to auto-normalise"
            )
        return self

    # ---- properties ------------------------------------------------------

    @property
    def modulation_order(self) -> int:
        return int(self.points.shape[0])

    @property
    def bits_per_symbol(self) -> int:
        return int(self.bit_labels.shape[1])

    @property
    def is_uniform(self) -> bool:
        return self.probabilities is None

    # ---- factories -------------------------------------------------------

    @classmethod
    def from_square_qam(cls, modulation_order: int) -> ConstellationConfig:
        """Build the Gray-coded square-QAM config, bit-compatible with
        :func:`hpcom.modulation.bits_to_symbols`."""
        points, bit_labels = square_qam(modulation_order)
        return cls(points=points, bit_labels=bit_labels)

    @classmethod
    def from_points(
        cls,
        points: ComplexArray,
        bit_labels: npt.NDArray[np.uint8] | None = None,
        probabilities: npt.NDArray[np.float64] | None = None,
    ) -> ConstellationConfig:
        """Build a config from arbitrary points (geometric shaping).

        Auto-normalises ``points`` to unit expected power. If ``bit_labels``
        is None, assigns labels by index in MSB-first binary — valid but not
        Gray-optimal; pass explicit labels for best BER.
        """
        points = np.asarray(points, dtype=np.complex128)
        m = points.shape[0]
        if points.ndim != 1 or not _is_power_of_two(m):
            raise ValueError(
                f"points must be 1-D with power-of-2 length, got shape {points.shape}"
            )

        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if probabilities.shape != (m,):
                raise ValueError(
                    f"probabilities must have shape ({m},), got {probabilities.shape}"
                )
            if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
                raise ValueError("probabilities must be non-negative and sum to 1")

        power = _expected_power(points, probabilities)
        if power <= 0:
            raise ValueError("constellation has zero expected power")
        points = points / np.sqrt(power)

        if bit_labels is None:
            k = int(np.log2(m))
            bit_labels = _default_bit_labels(m, k)

        return cls(points=points, bit_labels=bit_labels, probabilities=probabilities)

    # ---- mapping ---------------------------------------------------------

    def map(self, bits: npt.NDArray[np.uint8]) -> ComplexArray:
        """Map a flat bit stream to symbols using this constellation."""
        k = self.bits_per_symbol
        bits = np.asarray(bits, dtype=np.uint8)
        if bits.size % k != 0:
            raise ValueError(
                f"bits length {bits.size} is not a multiple of {k}"
            )
        groups = bits.reshape(-1, k)
        packed = _pack_bit_groups_msb_first(groups)

        label_keys = _pack_bit_groups_msb_first(self.bit_labels)
        lut = np.empty(self.modulation_order, dtype=np.complex128)
        lut[label_keys] = self.points
        return lut[packed]

    def demap(self, received: ComplexArray) -> npt.NDArray[np.uint8]:
        """Hard-decision demap. Ignores ``probabilities`` (uniform-prior decision)."""
        idx = nearest_symbol_indices(np.asarray(received, dtype=np.complex128), self.points)
        return self.bit_labels[idx].ravel()


def _default_bit_labels(m: int, k: int) -> npt.NDArray[np.uint8]:
    """Identity labels: row ``i`` → MSB-first ``k``-bit binary of ``i``."""
    out = np.empty((m, k), dtype=np.uint8)
    for i in range(m):
        for b in range(k):
            out[i, k - 1 - b] = (i >> b) & 1
    return out


def apsk(
    ring_sizes: tuple[int, ...] | list[int],
    ring_radii: tuple[float, ...] | list[float],
    ring_phases: tuple[float, ...] | list[float] | None = None,
) -> ConstellationConfig:
    """Amplitude-phase shift keying constellation: concentric rings.

    Each ring carries ``ring_sizes[i]`` equally-spaced points at radius
    ``ring_radii[i]``, optionally rotated by ``ring_phases[i]`` (radians).
    Total constellation size ``M = sum(ring_sizes)`` must be a power of 2.

    Points are laid out ring-by-ring starting from the innermost. Within
    each ring, points are emitted at angles ``ring_phases[i] + 2π k / M_i``
    for ``k = 0..M_i-1``.

    .. warning::
        Bit labels use MSB-first binary indexing (label[i] = binary(i)),
        **not** Gray coding. BER simulations over AWGN will be suboptimal
        without custom Gray labels. Pass explicit ``bit_labels`` to
        :meth:`ConstellationConfig.from_points` when BER accuracy matters.

    Examples
    --------
    Canonical DVB-S2 16-APSK (4 inner + 12 outer):

    >>> cfg = apsk(ring_sizes=(4, 12), ring_radii=(1.0, 2.85))
    >>> cfg.modulation_order
    16
    """
    if len(ring_sizes) != len(ring_radii):
        raise ValueError(
            f"ring_sizes and ring_radii must have the same length, "
            f"got {len(ring_sizes)} and {len(ring_radii)}"
        )
    if ring_phases is None:
        ring_phases = [0.0] * len(ring_sizes)
    elif len(ring_phases) != len(ring_sizes):
        raise ValueError(
            f"ring_phases must have length {len(ring_sizes)}, "
            f"got {len(ring_phases)}"
        )
    if any(n < 1 for n in ring_sizes):
        raise ValueError(f"all ring_sizes must be >= 1, got {ring_sizes}")
    if any(r <= 0 for r in ring_radii):
        raise ValueError(f"all ring_radii must be > 0, got {ring_radii}")

    total = int(sum(ring_sizes))
    if not _is_power_of_two(total):
        raise ValueError(
            f"sum(ring_sizes) must be a power of 2, got {total} "
            f"(ring_sizes={ring_sizes})"
        )

    points_list: list[complex] = []
    for n_ring, radius, phase in zip(ring_sizes, ring_radii, ring_phases, strict=True):
        angles = phase + 2.0 * np.pi * np.arange(n_ring) / n_ring
        ring = radius * np.exp(1j * angles)
        points_list.extend(ring.tolist())

    points = np.array(points_list, dtype=np.complex128)
    return ConstellationConfig.from_points(points)


__all__ = ["ConstellationConfig", "apsk", "bits_per_symbol"]
