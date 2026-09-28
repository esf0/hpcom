r"""Wavelength-division multiplexing: grid, transmitter, receiver.

A :class:`WDMConfig` is a uniform grid of ``n_channels`` channels spaced by
``spacing_hz`` and centred on the carrier (baseband 0 Hz):

.. math:: f_i = \left(i - \tfrac{N-1}{2}\right)\Delta f, \qquad i = 0, …, N-1.

(For odd ``N`` the centre channel sits at 0 Hz.) Frequencies are baseband
offsets in NumPy's FFT convention; with hpcom's NLSE convention a positive
offset corresponds to a *lower* optical frequency (see :doc:`/physics`).

:class:`WDMTransmitter` builds one RRC-shaped stream per channel (and
polarisation), shifts stream ``i`` by :math:`e^{2\pi i f_i t}` and sums them.
:class:`WDMReceiver` compensates dispersion on the composite field, then
shifts each selected channel back to baseband and matched-filters it — or,
with ``compensation=`` (e.g. digital back-propagation), shifts each channel
down first and runs the stage per channel on band-limited, resampled
samples.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import HpcomModel, Polymorphic
from hpcom._typing import ComplexArray
from hpcom.receiver import PulseShapedReceiver, ReceivedSymbols
from hpcom.transmitter import PulseShapedTransmitter, TransmittedSignal, Transmitter

__all__ = ["WDMConfig", "WDMReceivedSymbols", "WDMReceiver", "WDMSignal", "WDMTransmitter"]


class WDMConfig(HpcomModel):
    """Uniform WDM grid of ``n_channels`` channels at constant ``spacing_hz``."""

    n_channels: int = Field(
        ge=2, description="Number of channels (>= 2; use SingleCarrier* for one)."
    )
    spacing_hz: float = Field(gt=0, description="Centre-to-centre channel spacing in Hz.")

    def channel_frequencies_hz(self) -> npt.NDArray[np.float64]:
        """Baseband centre frequency of every channel, ``(i − (N−1)/2)·Δf``."""
        freqs: npt.NDArray[np.float64] = (
            np.arange(self.n_channels, dtype=np.float64) - (self.n_channels - 1) / 2.0
        ) * self.spacing_hz
        return freqs

    @property
    def total_bandwidth_hz(self) -> float:
        """Outer-to-outer channel-centre span ``(N−1)·Δf`` (excludes the channel bandwidth)."""
        return (self.n_channels - 1) * self.spacing_hz


def _check_grid(tx: PulseShapedTransmitter, wdm: WDMConfig) -> None:
    """Validate that the sampled band holds the grid and (if circular) is periodic."""
    fs = tx.symbol_rate_hz * tx.samples_per_symbol
    edge = abs(wdm.channel_frequencies_hz()[0]) + tx.symbol_rate_hz * (1.0 + tx.roll_off) / 2.0
    if fs < 2.0 * edge:
        raise ValueError(
            f"sample rate {fs:.4g} Hz cannot hold {wdm.n_channels} channels spaced "
            f"{wdm.spacing_hz:.4g} Hz (band edge ±{edge:.4g} Hz): increase samples_per_symbol "
            f"to at least {math.ceil(2.0 * edge / tx.symbol_rate_hz)}"
        )
    if tx.convolution != "circular":
        return
    # Periodicity: every carrier must complete an integer number of cycles in
    # the n_symbols/R_s window, i.e. lie on the DFT grid of the signal.
    half = 1 if wdm.n_channels % 2 else 2
    cycles = wdm.spacing_hz * tx.n_symbols / (half * tx.symbol_rate_hz)
    if abs(cycles - round(cycles)) > 1e-6 * max(1.0, abs(cycles)):
        ratio = Fraction(wdm.spacing_hz / (half * tx.symbol_rate_hz)).limit_denominator(10**6)
        raise ValueError(
            f"channel carriers are not periodic over the {tx.n_symbols}-symbol window "
            f"(spacing·T = {cycles:.6g} cycles is not an integer), so the circular "
            f"(periodic) signal would be discontinuous. Use n_symbols that is a multiple of "
            f"{ratio.denominator}, or convolution='linear'."
        )


class WDMSignal(TransmittedSignal):
    """Output of :class:`WDMTransmitter`.

    ``field``: composite, ``(n_samples,)`` or ``(2, n_samples)``.
    ``symbols``/``bits``: per channel, ``(n_channels, n)`` or ``(2, n_channels, n)``.
    """

    transmitter: Polymorphic[WDMTransmitter]

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        return self.transmitter.data_indices

    @property
    def channel_frequencies_hz(self) -> npt.NDArray[np.float64]:
        return self.transmitter.wdm.channel_frequencies_hz()


class WDMTransmitter(PulseShapedTransmitter):
    """WDM transmitter: ``wdm.n_channels`` independent RRC-QAM channels on a grid.

    All channels share the modulation, pulse, pilots and launch power
    (``launch_power_per_channel_dbm`` is per channel; ``launch_power_total_dbm``
    is split equally). ``samples_per_symbol`` sets the *composite* sample
    rate and must cover the whole grid.
    """

    wdm: WDMConfig

    @model_validator(mode="after")
    def _validate_wdm(self) -> WDMTransmitter:
        _check_grid(self, self.wdm)
        return self

    @property
    def n_channels(self) -> int:
        return self.wdm.n_channels

    def matched_receiver(self, **overrides: Any) -> WDMReceiver:
        """A :class:`WDMReceiver` for this signal (all channels unless ``channels=`` is given)."""
        return WDMReceiver(**{**self._pulse_fields(), "wdm": self.wdm, **overrides})

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.ArrayLike | None = None,
        symbols: npt.ArrayLike | None = None,
    ) -> WDMSignal:
        out_bits, data = self._draw_data(rng, bits, symbols)
        slots = self._slot_stream(data)  # (*pol, n_ch, n_symbols)
        channels = self._shape_streams(slots)  # (*pol, n_ch, n_samples)
        t = np.arange(self.n_samples) * self.dt_s
        carriers = np.exp(2j * np.pi * self.wdm.channel_frequencies_hz()[:, None] * t)
        field = np.sum(channels * carriers, axis=-2)
        return WDMSignal(transmitter=self, field=field, symbols=slots, bits=out_bits)


class WDMReceivedSymbols(ReceivedSymbols):
    """Output of :class:`WDMReceiver`; the channel axis holds :attr:`channels` in order."""

    channels: tuple[int, ...]


class WDMReceiver(PulseShapedReceiver):
    """Demultiplexes and demodulates the selected WDM ``channels`` (all by default).

    Output shapes: ``(n_selected, n)`` single-pol, ``(2, n_selected, n)``
    dual-pol — also when a single channel is selected.
    """

    wdm: WDMConfig
    channels: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _validate_channels(self) -> WDMReceiver:
        if self.channels is not None:
            if not self.channels:
                raise ValueError("`channels` must be non-empty (or None for all channels)")
            bad = [c for c in self.channels if not 0 <= c < self.wdm.n_channels]
            if bad:
                raise ValueError(f"channels {bad} are outside [0, {self.wdm.n_channels})")
            if len(set(self.channels)) != len(self.channels):
                raise ValueError(f"channels must be unique, got {self.channels}")
        return self

    @property
    def selected_channels(self) -> tuple[int, ...]:
        return self.channels if self.channels is not None else tuple(range(self.wdm.n_channels))

    def compatibility_issues(self, transmitter: Transmitter) -> list[str]:
        if not isinstance(transmitter, WDMTransmitter):
            return [f"WDMReceiver cannot receive a {type(transmitter).__name__}"]
        issues = self._base_issues(transmitter) + self._pulse_issues(transmitter)
        if transmitter.wdm != self.wdm:
            issues.append(f"wdm: transmitter {transmitter.wdm!r}, receiver {self.wdm!r}")
        return issues

    def reference(
        self, sent: TransmittedSignal
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]]:
        sel = list(self.selected_channels)
        return (
            np.take(np.asarray(sent.bits), sel, axis=-2),
            np.take(np.asarray(sent.data_symbols()), sel, axis=-2),
        )

    def receive(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        reference: TransmittedSignal | None = None,
    ) -> WDMReceivedSymbols:
        x = self._compensate_dispersion(self._check_field(field), dt_s)
        ref = self.reference(reference)[1] if reference is not None else None
        out = self._receive_field(x, dt_s, ref)
        return WDMReceivedSymbols(receiver=self, channels=self.selected_channels, **out)

    def _baseband(
        self, field: npt.NDArray[np.complex128], dt_s: float
    ) -> npt.NDArray[np.complex128]:
        freqs = np.asarray(self._channel_offsets_hz())
        t = np.arange(field.shape[-1]) * dt_s
        downshift = np.exp(-2j * np.pi * freqs[:, None] * t)  # (n_sel, n)
        streams: npt.NDArray[np.complex128] = field[..., None, :] * downshift  # (*pol, n_sel, n)
        return streams

    def _channel_offsets_hz(self) -> tuple[float, ...]:
        freqs = self.wdm.channel_frequencies_hz()[list(self.selected_channels)]
        return tuple(float(f) for f in freqs)


WDMSignal.model_rebuild()
WDMReceivedSymbols.model_rebuild()
