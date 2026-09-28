"""Bit ↔ symbol mapping for square-QAM constellations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray
from hpcom.modulation.constellations import bits_per_symbol, square_qam


def _pack_bit_groups_msb_first(bit_groups: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    """Pack the last axis of ``bit_groups`` (k bits, MSB first) into ints."""
    k = bit_groups.shape[-1]
    weights = 1 << np.arange(k - 1, -1, -1, dtype=np.int64)
    packed: npt.NDArray[np.int64] = (bit_groups.astype(np.int64) * weights).sum(axis=-1)
    return packed


def bits_to_symbols(bits: npt.NDArray[np.uint8], modulation_order: int) -> ComplexArray:
    """Map a flat bit stream to QAM symbols.

    ``len(bits)`` must be a multiple of ``log₂ modulation_order``.
    """
    k = bits_per_symbol(modulation_order)
    if bits.size % k != 0:
        raise ValueError(
            f"bits length {bits.size} is not a multiple of {k} (bits per symbol for "
            f"order {modulation_order})"
        )

    constellation, bit_labels = square_qam(modulation_order)
    bit_groups = np.asarray(bits, dtype=np.uint8).reshape(-1, k)
    packed = _pack_bit_groups_msb_first(bit_groups)

    label_keys = _pack_bit_groups_msb_first(bit_labels)
    lut = np.empty(modulation_order, dtype=np.complex128)
    lut[label_keys] = constellation
    return lut[packed]


def nearest_symbol_indices(
    received: ComplexArray, constellation: ComplexArray
) -> npt.NDArray[np.intp]:
    """For each received sample, the index of the closest constellation point."""
    diffs = received[:, None] - constellation[None, :]
    idx: npt.NDArray[np.intp] = np.argmin(np.abs(diffs) ** 2, axis=1).astype(np.intp)
    return idx


def symbols_to_bits(received: ComplexArray, modulation_order: int) -> npt.NDArray[np.uint8]:
    """Hard-decision demap: each received sample → its k bits."""
    constellation, bit_labels = square_qam(modulation_order)
    idx = nearest_symbol_indices(received, constellation)
    return bit_labels[idx].ravel()
