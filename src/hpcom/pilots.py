"""Time-domain pilot symbols for pulse-shaped (single-carrier / WDM) signals.

A :class:`PilotConfig` replaces every ``period``-th symbol slot — positions
``0, period, 2·period, …`` — with the known value ``symbol``. Pilots carry no
information bits: a transmitter's ``bits`` cover the data slots only, and
receivers can use the pilots to estimate the channel gain and phase
(``equalize="pilot"``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import HpcomModel


class PilotConfig(HpcomModel):
    """Uniform pilot schedule: slot ``k`` is a pilot iff ``k % period == 0``.

    ``symbol`` is on the same scale as the unit-power constellation (the
    default ``1 + 0j`` has unit energy).
    """

    period: int = Field(ge=2, description="Pilot insertion period (>= 2).")
    symbol: complex = 1.0 + 0.0j

    def pilot_indices(self, n_symbols: int) -> npt.NDArray[np.intp]:
        """Slots that carry a pilot."""
        return np.arange(0, n_symbols, self.period, dtype=np.intp)

    def data_indices(self, n_symbols: int) -> npt.NDArray[np.intp]:
        """Slots that carry data."""
        idx = np.arange(n_symbols, dtype=np.intp)
        data: npt.NDArray[np.intp] = idx[idx % self.period != 0]
        return data

    def n_pilots(self, n_symbols: int) -> int:
        """Number of pilot slots, ``⌈n_symbols / period⌉``."""
        return -(-n_symbols // self.period)

    def n_data(self, n_symbols: int) -> int:
        """Number of data slots."""
        return n_symbols - self.n_pilots(n_symbols)
