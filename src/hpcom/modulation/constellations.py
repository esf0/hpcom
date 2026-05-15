"""Square-QAM constellations with Gray coding."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray

_SUPPORTED_QAM_ORDERS = (4, 16, 64, 256, 1024)


def bits_per_symbol(modulation_order: int) -> int:
    """Number of bits encoded per symbol (``log₂ M``)."""
    if modulation_order < 2 or (modulation_order & (modulation_order - 1)) != 0:
        raise ValueError(
            f"modulation_order must be a power of 2 ≥ 2, got {modulation_order}"
        )
    return int(np.log2(modulation_order))


def _gray_code(n: int) -> npt.NDArray[np.int64]:
    """Length-n Gray code sequence as an array of integers."""
    return np.array([i ^ (i >> 1) for i in range(n)], dtype=np.int64)


def square_qam(modulation_order: int) -> tuple[ComplexArray, npt.NDArray[np.uint8]]:
    """Gray-coded square QAM constellation, normalized to unit average power.

    Parameters
    ----------
    modulation_order
        One of {4, 16, 64, 256, 1024}.

    Returns
    -------
    constellation
        Shape ``(M,)`` complex symbols. ``constellation[i]`` carries the bit
        pattern in row ``i`` of ``bit_labels``.
    bit_labels
        Shape ``(M, k)`` ``uint8`` array where ``k = log₂ M``. Bit order is
        MSB first; the first ``k/2`` bits encode the Q-axis level, the last
        ``k/2`` encode the I-axis level. Adjacent constellation points along
        either axis differ by exactly one bit.
    """
    if modulation_order not in _SUPPORTED_QAM_ORDERS:
        raise ValueError(
            f"square_qam supports orders {_SUPPORTED_QAM_ORDERS}, got {modulation_order}"
        )

    side = round(float(np.sqrt(modulation_order)))
    bits_per_axis = round(float(np.log2(side)))
    k = 2 * bits_per_axis

    levels = np.arange(side) * 2 - (side - 1)  # e.g. [-3, -1, 1, 3] for 16-QAM
    gray = _gray_code(side)

    # E[|s|²] for unscaled levels = 2 · mean(level²) = 2 · (side²-1)/3.
    norm = float(np.sqrt(2.0 * (side**2 - 1) / 3.0))

    constellation = np.empty(modulation_order, dtype=np.complex128)
    bit_labels = np.empty((modulation_order, k), dtype=np.uint8)

    for q_idx in range(side):
        for i_idx in range(side):
            sym_idx = q_idx * side + i_idx
            constellation[sym_idx] = (levels[i_idx] + 1j * levels[q_idx]) / norm
            q_bits = [(int(gray[q_idx]) >> b) & 1 for b in range(bits_per_axis - 1, -1, -1)]
            i_bits = [(int(gray[i_idx]) >> b) & 1 for b in range(bits_per_axis - 1, -1, -1)]
            bit_labels[sym_idx] = np.array(q_bits + i_bits, dtype=np.uint8)

    return constellation, bit_labels
