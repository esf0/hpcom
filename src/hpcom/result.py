"""Output container for a :class:`hpcom.simulation.Simulation` run."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from hpcom._typing import ComplexArray
from hpcom.metrics import Metrics


class SimulationMetadata(BaseModel):
    """Reproducibility + provenance for a single run."""

    model_config = ConfigDict(frozen=True)

    seed: int | None
    runtime_s: float
    config: dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    """End-to-end simulation output.

    Carries enough state for the user to inspect signals, recompute custom
    metrics, or re-run with the same config.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    received_field: ComplexArray
    decoded_symbols: ComplexArray
    decoded_bits: npt.NDArray[np.uint8]
    metrics: Metrics
    metadata: SimulationMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary (arrays excluded; use ``np.savez`` for those)."""
        return {
            "metrics": self.metrics.model_dump(),
            "metadata": {
                "seed": self.metadata.seed,
                "runtime_s": self.metadata.runtime_s,
                "config": self.metadata.config,
            },
            "n_samples": int(self.received_field.size),
            "n_decoded_symbols": int(self.decoded_symbols.size),
        }

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"Result(ber={m.ber:.3e}, evm_rms={m.evm_rms:.3e}, "
            f"n_bits={m.n_bits}, runtime={self.metadata.runtime_s:.3f}s)"
        )
