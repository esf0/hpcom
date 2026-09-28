"""Constellation selection shared by transmitters and receivers."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import model_validator

from hpcom._model import HpcomModel
from hpcom.modulation.dm import ConstantCompositionDM, empirical_symbol_energy
from hpcom.modulation.shaping import ConstellationConfig


class ModulationSpec(HpcomModel):
    """Which symbols are sent and how bits map onto them.

    Exactly one of:

    * ``modulation_order`` — Gray-coded square QAM of that order, or
    * ``constellation`` — any :class:`~hpcom.modulation.ConstellationConfig`
      (geometric shaping; with ``probabilities``, probabilistic shaping).

    ``dm`` (optional) is a :class:`~hpcom.modulation.ConstantCompositionDM`
    whose ``target_pmf`` equals ``constellation.probabilities``. Without a DM,
    a constellation that has ``probabilities`` is sampled i.i.d. from them and
    the "bits" are the bit labels of the drawn symbols.
    """

    modulation_order: int | None = None
    constellation: ConstellationConfig | None = None
    dm: ConstantCompositionDM | None = None

    _exclusive_fields: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("modulation_order", "constellation"),
    )

    @model_validator(mode="after")
    def _validate_modulation(self) -> ModulationSpec:
        if (self.modulation_order is None) == (self.constellation is None):
            raise ValueError("set exactly one of `modulation_order` or `constellation`")
        if self.modulation_order is not None:
            ConstellationConfig.from_square_qam(self.modulation_order)  # validates the order
        if self.dm is not None:
            cfg = self.constellation
            if cfg is None or cfg.probabilities is None:
                raise ValueError("`dm` requires a `constellation` with `probabilities`")
            if self.dm.target_pmf.shape != cfg.probabilities.shape or not np.allclose(
                self.dm.target_pmf, cfg.probabilities, rtol=0, atol=1e-12
            ):
                raise ValueError("dm.target_pmf must equal constellation.probabilities")
        return self

    @property
    def resolved_constellation(self) -> ConstellationConfig:
        """The constellation in use (built from ``modulation_order`` if needed)."""
        if self.constellation is not None:
            return self.constellation
        assert self.modulation_order is not None
        return ConstellationConfig.from_square_qam(self.modulation_order)

    @property
    def effective_modulation_order(self) -> int:
        """Constellation size ``M`` (defined whichever field was used)."""
        return self.resolved_constellation.modulation_order

    @property
    def bits_per_symbol(self) -> int:
        """``log₂ M`` (label bits per symbol)."""
        return self.resolved_constellation.bits_per_symbol

    @property
    def data_symbol_energy(self) -> float:
        """Expected ``|s|²`` of transmitted data symbols.

        1 by construction, except with a DM, whose exact emitted distribution
        (the rounded composition) can differ slightly from the target PMF.
        """
        cfg = self.resolved_constellation
        if self.dm is not None:
            return empirical_symbol_energy(cfg.points, self.dm.empirical_pmf)
        return empirical_symbol_energy(cfg.points, cfg.pmf)

    def modulation_mismatch(self, other: ModulationSpec) -> str | None:
        """Describe how ``other`` maps bits differently, or ``None`` if identical."""
        if self.resolved_constellation != other.resolved_constellation:
            return "constellations differ (points, labels or probabilities)"
        if self.dm != other.dm:
            return "distribution matchers differ"
        return None
