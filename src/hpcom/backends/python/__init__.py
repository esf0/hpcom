"""Pure-Python (NumPy) implementations of the Tx/Rx hot-path primitives.

This namespace re-exports the canonical pure-Python implementations from their
home modules (:mod:`hpcom.modulation`, :mod:`hpcom._pulse`). It exists so a
future ``hpcom.backends.rust`` sibling can mirror the same symbol surface and
a top-level selector (``backend="python" | "rust"``) can swap them transparently
without breaking any user-facing import.

Today there is exactly one backend (Python). The selector is intentionally
absent — adding it before a second backend exists would be designing against a
hypothetical requirement. When the Rust backend lands, this module's docstring
gets updated and the dispatch logic goes in :mod:`hpcom.backends`.
"""

from __future__ import annotations

from hpcom._pulse import root_raised_cosine
from hpcom.modulation import (
    bits_per_symbol,
    bits_to_symbols,
    nearest_symbol_indices,
    square_qam,
    symbols_to_bits,
)

__all__ = [
    "bits_per_symbol",
    "bits_to_symbols",
    "nearest_symbol_indices",
    "root_raised_cosine",
    "square_qam",
    "symbols_to_bits",
]
