"""Modulation: constellations and symbol/bit mapping.

Pure-NumPy, framework-free. Square-QAM with Gray coding via the legacy-style
flat API (``bits_to_symbols`` / ``symbols_to_bits``), plus a richer
:class:`ConstellationConfig` for geometric and probabilistic shaping.
"""

from hpcom.modulation.constellations import bits_per_symbol, square_qam
from hpcom.modulation.mapping import (
    bits_to_symbols,
    nearest_symbol_indices,
    symbols_to_bits,
)
from hpcom.modulation.shaping import ConstellationConfig

__all__ = [
    "ConstellationConfig",
    "bits_per_symbol",
    "bits_to_symbols",
    "nearest_symbol_indices",
    "square_qam",
    "symbols_to_bits",
]
