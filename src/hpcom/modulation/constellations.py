"""Square-QAM constellations with Gray coding."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import numpy.typing as npt

from hpcom._model import readonly

#: Square-QAM orders supported by :func:`square_qam`.
SUPPORTED_QAM_ORDERS = (4, 16, 64, 256, 1024, 4096)


def bits_per_symbol(modulation_order: int) -> int:
    """Number of bits per symbol, ``log₂ M``."""
    if modulation_order < 2 or (modulation_order & (modulation_order - 1)) != 0:
        raise ValueError(f"modulation_order must be a power of 2 ≥ 2, got {modulation_order}")
    return int(modulation_order).bit_length() - 1


@cache
def square_qam(
    modulation_order: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.uint8]]:
    """Gray-coded square M-QAM, normalised to unit average power.

    Returns
    -------
    points
        ``(M,)`` complex symbols; row ``i`` of ``bit_labels`` is the label of
        ``points[i]``. Point ``i = q·√M + p`` has in-phase level index ``p``
        and quadrature level index ``q`` (levels ``−(√M−1), …, √M−1``, step 2).
    bit_labels
        ``(M, log₂M)`` uint8, MSB first. The first half of each label is the
        Gray code of the Q level, the second half that of the I level, so
        nearest neighbours along either axis differ in exactly one bit.

    Both arrays are read-only (the result is cached and shared).
    """
    if modulation_order not in SUPPORTED_QAM_ORDERS:
        raise ValueError(
            f"square_qam supports orders {SUPPORTED_QAM_ORDERS}, got {modulation_order}"
        )
    side = math.isqrt(modulation_order)
    bits_per_axis = bits_per_symbol(side)
    levels = 2.0 * np.arange(side) - (side - 1)
    norm = np.sqrt(2.0 * (side**2 - 1) / 3.0)  # RMS of the unscaled grid
    gray = np.arange(side) ^ (np.arange(side) >> 1)
    axis_bits = (gray[:, None] >> np.arange(bits_per_axis - 1, -1, -1)) & 1  # (side, bpa)

    q_idx, i_idx = np.divmod(np.arange(modulation_order), side)
    points = (levels[i_idx] + 1j * levels[q_idx]) / norm
    labels = np.concatenate([axis_bits[q_idx], axis_bits[i_idx]], axis=1).astype(np.uint8)
    return readonly(points, np.complex128), readonly(labels, np.uint8)
