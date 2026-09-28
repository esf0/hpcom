"""Bit ↔ symbol mapping helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray
from hpcom.backends.kernels import get_kernels
from hpcom.modulation.constellations import bits_per_symbol, square_qam


def pack_bits_msb_first(bit_groups: npt.NDArray[np.uint8]) -> npt.NDArray[np.intp]:
    """Pack the last axis of ``bit_groups`` (k bits, MSB first) into integers."""
    k = bit_groups.shape[-1]
    weights = 1 << np.arange(k - 1, -1, -1, dtype=np.intp)
    packed: npt.NDArray[np.intp] = (bit_groups.astype(np.intp) * weights).sum(axis=-1)
    return packed


def nearest_symbol_indices(received: ComplexArray, points: ComplexArray) -> npt.NDArray[np.intp]:
    """Index of the nearest constellation point for every sample of ``received``.

    Works for any input shape (the result has the same shape). Memory use is
    bounded independently of ``len(received) × len(points)``.
    """
    rx: npt.NDArray[np.complex128] = np.asarray(received, dtype=np.complex128)
    pts: npt.NDArray[np.complex128] = np.asarray(points, dtype=np.complex128)
    flat = get_kernels().nearest_indices(rx.reshape(-1), pts)
    return flat.reshape(rx.shape)


def bits_to_symbols(bits: npt.NDArray[np.uint8], modulation_order: int) -> ComplexArray:
    """Map a flat bit stream to Gray-coded square-QAM symbols (unit average power)."""
    from hpcom.modulation.shaping import ConstellationConfig

    return ConstellationConfig.from_square_qam(modulation_order).map(bits)


def symbols_to_bits(received: ComplexArray, modulation_order: int) -> npt.NDArray[np.uint8]:
    """Hard-decision demap of received samples to square-QAM bits (flat, MSB first)."""
    bits_per_symbol(modulation_order)
    points, labels = square_qam(modulation_order)
    idx = nearest_symbol_indices(np.asarray(received).reshape(-1), points)
    return labels[idx].reshape(-1)
