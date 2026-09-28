"""Modulation: constellations, bit mapping, probabilistic shaping.

* :func:`square_qam`, :class:`ConstellationConfig`, :func:`apsk` — constellations
  (Gray-coded square QAM; arbitrary geometric shaping).
* :func:`maxwell_boltzmann`, :func:`maxwell_boltzmann_for_entropy`,
  :class:`ConstantCompositionDM` — probabilistic shaping.
* :func:`bits_to_symbols`, :func:`symbols_to_bits`,
  :func:`nearest_symbol_indices` — flat helpers.
"""

from hpcom.modulation.constellations import (
    DEFAULT_QAM_LABELING,
    SquareQAMLabeling,
    bits_per_symbol,
    square_qam,
)
from hpcom.modulation.dm import (
    ConstantCompositionDM,
    composition_from_pmf,
    maxwell_boltzmann,
    maxwell_boltzmann_for_entropy,
)
from hpcom.modulation.mapping import bits_to_symbols, nearest_symbol_indices, symbols_to_bits
from hpcom.modulation.shaping import ConstellationConfig, apsk, natural_labels
from hpcom.modulation.spec import ModulationSpec

__all__ = [
    "DEFAULT_QAM_LABELING",
    "ConstantCompositionDM",
    "ConstellationConfig",
    "ModulationSpec",
    "SquareQAMLabeling",
    "apsk",
    "bits_per_symbol",
    "bits_to_symbols",
    "composition_from_pmf",
    "maxwell_boltzmann",
    "maxwell_boltzmann_for_entropy",
    "natural_labels",
    "nearest_symbol_indices",
    "square_qam",
    "symbols_to_bits",
]
