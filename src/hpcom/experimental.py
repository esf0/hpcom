"""Template for prototyping a custom waveform.

:class:`ExperimentalTransmission` / :class:`ExperimentalReceiver` are a
minimal, complete transmitter/receiver pair — rectangular-pulse QAM
(sample-and-hold) — that plugs into everything else in hpcom:
:class:`~hpcom.link.Link`, :class:`~hpcom.simulation.Simulation`, metrics,
polarisation handling, launch power, distribution matching. Copy this module,
rename the classes, and replace the blocks marked ``>>> REPLACE``.

The waveform itself is deliberately poor (sinc spectrum, no matched filter):
it survives back-to-back and dispersion-compensated links, nothing more.

Contract for a new waveform
---------------------------
A transmitter subclasses :class:`~hpcom.transmitter.Transmitter` and provides

1. ``dt_s`` — sample period of the field;
2. ``n_data_symbols_per_stream`` — data symbols per polarisation (per channel);
3. ``generate(*, rng, bits, symbols)`` — obtain data with
   :meth:`~hpcom.transmitter.Transmitter._draw_data`, build the field in √W
   (scale unit-energy symbols by ``sqrt(launch_power_per_polarization_w)``)
   and return a :class:`~hpcom.transmitter.TransmittedSignal` subclass.

A receiver subclasses :class:`~hpcom.receiver.Receiver` and provides
``receive`` (demodulate to symbol-rate samples, then call
:meth:`~hpcom.receiver.Receiver._detect`), ``reference`` (the transmitted
bits/symbols aligned with its output) and ``compatibility_issues``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import Polymorphic
from hpcom._typing import ComplexArray
from hpcom.receiver import ReceivedSymbols, Receiver
from hpcom.transmitter import TransmittedSignal, Transmitter

__all__ = [
    "ExperimentalReceivedSymbols",
    "ExperimentalReceiver",
    "ExperimentalSignal",
    "ExperimentalTransmission",
]


class ExperimentalSignal(TransmittedSignal):
    """Output of :class:`ExperimentalTransmission`; add waveform-specific metadata here."""

    transmitter: Polymorphic[ExperimentalTransmission]

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        return np.arange(self.symbols.shape[-1], dtype=np.intp)


class ExperimentalTransmission(Transmitter):
    """Rectangular-pulse QAM: each symbol held for ``samples_per_symbol`` samples."""

    symbol_rate_hz: float = Field(gt=0)
    n_symbols: int = Field(ge=1)
    samples_per_symbol: int = Field(default=8, ge=1)

    @property
    def dt_s(self) -> float:
        # >>> REPLACE: the sample period of your field.
        return 1.0 / (self.symbol_rate_hz * self.samples_per_symbol)

    @property
    def n_data_symbols_per_stream(self) -> int:
        return self.n_symbols

    def matched_receiver(self, **overrides: Any) -> ExperimentalReceiver:
        # >>> REPLACE: copy your waveform's settings into its receiver.
        return ExperimentalReceiver(
            **{
                **self._modulation_fields(),
                "samples_per_symbol": self.samples_per_symbol,
                **overrides,
            }
        )

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.ArrayLike | None = None,
        symbols: npt.ArrayLike | None = None,
    ) -> ExperimentalSignal:
        out_bits, data = self._draw_data(rng, bits, symbols)  # (*stream_shape, n)
        # >>> REPLACE: build the field from `data` (unit-energy symbols).
        field = np.repeat(data, self.samples_per_symbol, axis=-1)
        field = field * np.sqrt(self.launch_power_per_polarization_w / self.data_symbol_energy)
        return ExperimentalSignal(transmitter=self, field=field, symbols=data, bits=out_bits)


class ExperimentalReceivedSymbols(ReceivedSymbols):
    """Output of :class:`ExperimentalReceiver`."""


class ExperimentalReceiver(Receiver):
    """Samples the centre of every symbol period, then equalises and decides."""

    samples_per_symbol: int = Field(default=8, ge=1)

    def compatibility_issues(self, transmitter: Transmitter) -> list[str]:
        if not isinstance(transmitter, ExperimentalTransmission):
            return [f"ExperimentalReceiver cannot receive a {type(transmitter).__name__}"]
        issues = self._base_issues(transmitter)
        if transmitter.samples_per_symbol != self.samples_per_symbol:
            issues.append("samples_per_symbol differs")
        return issues

    def reference(
        self, sent: TransmittedSignal
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]]:
        return np.asarray(sent.bits), np.asarray(sent.symbols)

    def receive(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        reference: TransmittedSignal | None = None,
    ) -> ExperimentalReceivedSymbols:
        x = self._compensate_dispersion(self._check_field(field), dt_s)
        # >>> REPLACE: recover one sample per symbol (matched filter, timing, …).
        raw = x[..., self.samples_per_symbol // 2 :: self.samples_per_symbol]
        ref = self.reference(reference)[1] if reference is not None else None
        return ExperimentalReceivedSymbols(receiver=self, **self._detect(raw, reference_data=ref))


ExperimentalSignal.model_rebuild()
ExperimentalReceivedSymbols.model_rebuild()
