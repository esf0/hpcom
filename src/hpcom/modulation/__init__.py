"""Modulation: constellations and symbol/bit mapping.

Pure-NumPy, framework-free. Square-QAM with Gray coding only in v1; PSK and
APSK can be added later behind the same :func:`bits_to_symbols` /
:func:`symbols_to_bits` interface.
"""

from hpcom.modulation.constellations import bits_per_symbol, square_qam
from hpcom.modulation.mapping import (
    bits_to_symbols,
    nearest_symbol_indices,
    symbols_to_bits,
)

__all__ = [
    "bits_per_symbol",
    "bits_to_symbols",
    "nearest_symbol_indices",
    "square_qam",
    "symbols_to_bits",
]
