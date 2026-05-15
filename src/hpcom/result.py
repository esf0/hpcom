"""Output container for a :class:`hpcom.simulation.Simulation` run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray
from hpcom.metrics import Metrics


@dataclass(frozen=True)
class SimulationMetadata:
    """Reproducibility + provenance for a single run."""

    seed: int | None
    runtime_s: float
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    """End-to-end simulation output.

    Carries enough state for the user to inspect signals, recompute custom
    metrics, or re-run with the same config.
    """

    received_field: ComplexArray
    decoded_symbols: ComplexArray
    decoded_bits: npt.NDArray[np.uint8]
    metrics: Metrics
    metadata: SimulationMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary (arrays excluded; use ``np.savez`` for those)."""
        return {
            "metrics": asdict(self.metrics),
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
