"""WDM (wavelength-division multiplexing) configuration.

A :class:`WDMConfig` describes a uniform grid of ``n_channels`` channels with
constant ``spacing_hz`` between adjacent channel centers, centered on the
optical carrier (baseband 0 Hz). The transmitter superimposes per-channel
RRC-shaped baseband fields onto this grid; the receiver downshifts a single
selected channel back to baseband before its matched filter.

For ``n_channels = N`` and spacing :math:`\\Delta f`, the channel center
frequencies (Hz, baseband) are

.. math:: f_i = (i - (N-1)/2) \\cdot \\Delta f, \\quad i = 0, ..., N-1

so the grid straddles 0 Hz symmetrically (with a center channel exactly at 0
for odd ``N``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class WDMConfig:
    """Uniform WDM grid: ``n_channels`` channels at constant ``spacing_hz``."""

    n_channels: int
    spacing_hz: float

    def __post_init__(self) -> None:
        if self.n_channels < 2:
            raise ValueError(
                f"WDMConfig.n_channels must be >= 2 (use a plain Transmitter "
                f"for single-channel), got {self.n_channels}"
            )
        if self.spacing_hz <= 0:
            raise ValueError(
                f"WDMConfig.spacing_hz must be > 0, got {self.spacing_hz}"
            )

    def channel_frequencies_hz(self) -> npt.NDArray[np.float64]:
        """Per-channel center frequencies (Hz, baseband), centered on 0 Hz."""
        idx = np.arange(self.n_channels, dtype=np.float64)
        return (idx - (self.n_channels - 1) / 2.0) * self.spacing_hz

    @property
    def total_bandwidth_hz(self) -> float:
        """Outer-to-outer channel-center span, in Hz. (Channel bandwidth is
        added separately by the caller based on symbol rate and roll-off.)"""
        return float((self.n_channels - 1) * self.spacing_hz)
