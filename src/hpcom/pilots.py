"""Pilot-symbol configuration.

A :class:`PilotConfig` describes a uniform pilot pattern: every ``period``-th
symbol slot carries a known reference value (``symbol``) instead of a random
data symbol. Pilots are the foundation for downstream DSP — carrier-phase
estimation, carrier-frequency offset, per-channel gain — though no estimator
ships in v1. The Transmitter inserts pilots; the Receiver extracts them and
returns the raw post-matched-filter samples at pilot positions so users (or
later modules) can run their own estimators.

Pilot positions are computed as ``0, period, 2·period, …`` (offset 0) over
``n_symbols``. The number of pilot slots is ``ceil(n_symbols / period)``.

Two BER conventions are both supported, controlled by
:attr:`ReceivedSymbols.data_bits` vs :attr:`ReceivedSymbols.bits`:

- **Option α** (default in examples): score on ``data_bits`` only — the bit
  stream with pilot positions stripped out.
- **Option β**: score on full ``bits`` and mask externally — kept zero-filled
  at pilot positions on the Tx side so it round-trips structurally.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


class PilotConfig(BaseModel):
    """Uniform pilot schedule: every ``period``-th symbol is a reference."""

    model_config = ConfigDict(frozen=True)

    period: int = Field(ge=2, description="Pilot insertion period; must be >= 2.")
    symbol: complex = 1.0 + 0.0j

    def pilot_indices(self, n_symbols: int) -> npt.NDArray[np.intp]:
        """Symbol-grid positions where a pilot is transmitted."""
        return np.arange(0, n_symbols, self.period, dtype=np.intp)

    def data_indices(self, n_symbols: int) -> npt.NDArray[np.intp]:
        """Symbol-grid positions where a data symbol is transmitted."""
        mask = np.ones(n_symbols, dtype=bool)
        mask[self.pilot_indices(n_symbols)] = False
        return np.flatnonzero(mask).astype(np.intp)

    def n_pilots(self, n_symbols: int) -> int:
        """Count of pilot positions in an ``n_symbols``-long stream."""
        return int(np.ceil(n_symbols / self.period))

    def n_data(self, n_symbols: int) -> int:
        """Count of data positions in an ``n_symbols``-long stream."""
        return n_symbols - self.n_pilots(n_symbols)
