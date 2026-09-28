r"""Optical receivers: abstract base, pulse-shaped base, single-carrier QAM.

Mirrors :mod:`hpcom.transmitter`::

    Receiver (abstract)                          — CD compensation, equaliser, decisions
    ├── PulseShapedReceiver (abstract)           — RRC matched filter, pilots
    │   ├── SingleCarrierReceiver
    │   └── WDMReceiver      (hpcom.wdm)         — demultiplexes selected channels
    └── OFDMReceiver         (hpcom.ofdm)

Processing chain of every receiver:

1. **Chromatic-dispersion compensation** of the whole received field (before
   any channel selection): multiply the spectrum by
   :math:`\exp[-i\,\beta_2L\,\omega^2/2 + i\,\beta_3L\,\omega^3/6]`, the
   inverse of the fiber's dispersive phase (see :doc:`/physics`). Set
   ``cd_compensation_s2 = link.total_dispersion_s2`` (and ``_s3``).
2. **Waveform-specific demodulation** to symbol-rate samples (matched
   filter, or FFT for OFDM).
3. **Equalisation** by one complex gain ``g`` per stream (polarisation ×
   channel), ``y = raw / g``, chosen by ``equalize``:

   ``"auto"`` (default)
       ``"pilot"`` if the receiver has pilots, else ``"data_aided"`` if the
       transmitted signal is supplied (as :class:`~hpcom.simulation.Simulation`
       always does), else ``"unit_power"``. The mode actually used is
       reported in :attr:`ReceivedSymbols.equalizer`.
   ``"none"``
       ``g = 1`` — raw samples (in √W, including launch power and link gain).
   ``"unit_power"``
       ``g = sqrt(mean|raw|² / E_s)`` — blind amplitude normalisation to the
       constellation energy ``E_s``. Does not correct phase.
   ``"pilot"``
       Least-squares gain and phase from the known pilots,
       ``g = Σ p̄ r / Σ |p|²``.
   ``"data_aided"``
       Least-squares gain and phase against the *transmitted* data symbols
       (a genie; needs ``reference=`` in :meth:`Receiver.receive`). This is
       the classic removal of the mean nonlinear phase rotation used in
       fiber-nonlinearity studies (pre-v1 hpcom's ``nonlinear_shift``).
4. **Hard decision** to the nearest constellation point, then bit labels —
   or distribution-matcher decoding when ``dm`` is set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import (
    BoolArrayView,
    ComplexArrayView,
    HpcomModel,
    IndexArrayView,
    Polymorphic,
    UInt8ArrayView,
)
from hpcom._pulse import Convolution, matched_filter
from hpcom._typing import ComplexArray, Polarization
from hpcom.modulation.spec import ModulationSpec
from hpcom.pilots import PilotConfig
from hpcom.transmitter import (
    PulseShapedTransmitter,
    SingleCarrierTransmitter,
    TransmittedSignal,
    Transmitter,
)

#: An equaliser that is actually applied (``"auto"`` resolves to one of these).
ResolvedEqualizer = Literal["none", "unit_power", "pilot", "data_aided"]
#: Per-stream complex-gain equaliser selection (see the module docstring).
EqualizerMode = Literal["auto", "none", "unit_power", "pilot", "data_aided"]


class Detection(TypedDict):
    """Fields of :class:`ReceivedSymbols` produced by :meth:`Receiver._detect`."""

    symbols: npt.NDArray[np.complex128]
    decisions: npt.NDArray[np.intp]
    bits: npt.NDArray[np.uint8]
    gain: npt.NDArray[np.complex128]
    equalizer: ResolvedEqualizer
    pilot_symbols: npt.NDArray[np.complex128] | None
    dm_block_valid: npt.NDArray[np.bool_] | None


__all__ = [
    "EqualizerMode",
    "PulseShapedReceiver",
    "ReceivedSymbols",
    "Receiver",
    "ResolvedEqualizer",
    "SingleCarrierReceivedSymbols",
    "SingleCarrierReceiver",
]


# ---------------------------------------------------------------------------
# Received symbols
# ---------------------------------------------------------------------------


class ReceivedSymbols(HpcomModel):
    """Output of :meth:`Receiver.receive` (one subclass per waveform).

    Leading axes follow the transmitter's ``stream_shape`` (for WDM: the
    *selected* channels).

    Attributes
    ----------
    receiver
        The configuration that produced this output.
    symbols
        Equalised data symbols, ``(*streams, n_data)``.
    decisions
        Index of the decided constellation point for every data symbol.
    bits
        Decoded information bits, ``(*streams, n_bits)``.
    gain
        Complex gain removed per stream (``symbols = raw / gain``), ``(*streams,)``.
        With a pilot or data-aided equaliser its phase is the mean (e.g.
        nonlinear) phase rotation of the link.
    equalizer
        The equaliser that was applied (``"auto"`` resolved).
    pilot_symbols
        Equalised received pilots, ``(*streams, n_pilots)``, when pilots exist.
    dm_block_valid
        With a DM: whether each received block was a valid codeword,
        ``(*streams, n_blocks)``. Invalid blocks decode to zero bits.
    """

    receiver: Polymorphic[Receiver]
    symbols: ComplexArrayView
    decisions: IndexArrayView
    bits: UInt8ArrayView
    gain: ComplexArrayView
    equalizer: ResolvedEqualizer
    pilot_symbols: ComplexArrayView | None = None
    dm_block_valid: BoolArrayView | None = None

    @property
    def polarization(self) -> Polarization:
        return self.receiver.polarization


class SingleCarrierReceivedSymbols(ReceivedSymbols):
    """Output of :class:`SingleCarrierReceiver`."""


# ---------------------------------------------------------------------------
# Receiver base
# ---------------------------------------------------------------------------


class Receiver(ModulationSpec, ABC):
    """Abstract receiver (see the module docstring for the processing chain).

    Attributes
    ----------
    polarization
        Must match the transmitter.
    equalize
        Per-stream complex-gain equaliser: ``"auto"`` (default), ``"none"``,
        ``"unit_power"``, ``"pilot"`` or ``"data_aided"`` (module docstring).
    cd_compensation_s2
        Accumulated ``Σ β₂L`` (s²) to remove; ``link.total_dispersion_s2``.
    cd_compensation_s3
        Accumulated ``Σ β₃L`` (s³) to remove; ``link.total_beta3_s3``.
    """

    polarization: Polarization = "single"
    equalize: EqualizerMode = "auto"
    cd_compensation_s2: float = 0.0
    cd_compensation_s3: float = 0.0

    @abstractmethod
    def receive(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        reference: TransmittedSignal | None = None,
    ) -> ReceivedSymbols:
        """Demodulate ``field`` (sample period ``dt_s``).

        ``reference`` (the transmitted signal) is used only by the data-aided
        equaliser (``equalize="data_aided"``, or ``"auto"`` without pilots).
        """

    @abstractmethod
    def compatibility_issues(self, transmitter: Transmitter) -> list[str]:
        """Reasons this receiver cannot demodulate ``transmitter``'s signal (empty if none)."""

    @abstractmethod
    def reference(
        self, sent: TransmittedSignal
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]]:
        """Transmitted ``(bits, data_symbols)`` aligned with :meth:`receive`'s output."""

    @property
    def transient_symbols(self) -> int:
        """Data symbols at each block edge affected by filter transients."""
        return 0

    # ---- shared helpers ----------------------------------------------------

    def _base_issues(self, transmitter: Transmitter) -> list[str]:
        issues = []
        if transmitter.polarization != self.polarization:
            issues.append(
                f"polarization: transmitter {transmitter.polarization!r}, "
                f"receiver {self.polarization!r}"
            )
        mismatch = self.modulation_mismatch(transmitter)
        if mismatch:
            issues.append(mismatch)
        return issues

    def _check_field(self, field: ComplexArray) -> npt.NDArray[np.complex128]:
        x = np.asarray(field, dtype=np.complex128)
        if self.polarization == "single" and x.ndim != 1:
            raise ValueError(f"single-polarisation receiver expects a 1-D field, got {x.shape}")
        if self.polarization == "dual" and (x.ndim != 2 or x.shape[0] != 2):
            raise ValueError(f"dual-polarisation receiver expects shape (2, n), got {x.shape}")
        return x

    def _compensate_dispersion(
        self, field: npt.NDArray[np.complex128], dt_s: float
    ) -> npt.NDArray[np.complex128]:
        if self.cd_compensation_s2 == 0.0 and self.cd_compensation_s3 == 0.0:
            return field
        omega = 2.0 * np.pi * np.fft.fftfreq(field.shape[-1], d=dt_s)
        phase = -self.cd_compensation_s2 / 2.0 * omega**2 + self.cd_compensation_s3 / 6.0 * omega**3
        out: npt.NDArray[np.complex128] = np.fft.ifft(
            np.fft.fft(field, axis=-1) * np.exp(1j * phase), axis=-1
        )
        return out

    def _detect(
        self,
        raw_data: npt.NDArray[np.complex128],
        *,
        raw_pilots: npt.NDArray[np.complex128] | None = None,
        pilot_values: npt.NDArray[np.complex128] | None = None,
        reference_data: npt.NDArray[np.complex128] | None = None,
        gain: npt.NDArray[np.complex128] | None = None,
        equalizer: ResolvedEqualizer | None = None,
    ) -> Detection:
        """Equalise, decide and decode ``(*streams, n)`` symbol-rate samples.

        ``gain`` (with the ``equalizer`` that produced it) overrides the
        estimate — used by receivers that equalise per subcarrier themselves.
        """
        has_pilots = (
            raw_pilots is not None and pilot_values is not None and raw_pilots.shape[-1] > 0
        )
        mode = equalizer or self._resolve_equalizer(has_pilots, reference_data is not None)
        if gain is None:
            gain = self._estimate_gain(mode, raw_data, raw_pilots, pilot_values, reference_data)
        data = raw_data / gain[..., None]
        pilots = None if raw_pilots is None else raw_pilots / gain[..., None]
        cfg = self.resolved_constellation
        decisions = cfg.nearest(data)
        lead, n = decisions.shape[:-1], decisions.shape[-1]
        flat = decisions.reshape(-1, n)
        dm_valid = None
        if self.dm is not None:
            decoded = [self.dm.decode_with_errors(row) for row in flat]
            bits = np.stack([b for b, _ in decoded])
            dm_valid = np.stack([v for _, v in decoded]).reshape(*lead, -1)
        else:
            bits = cfg.bit_labels[flat].reshape(flat.shape[0], -1)
        return Detection(
            symbols=data,
            decisions=decisions,
            bits=bits.reshape(*lead, -1),
            gain=gain,
            equalizer=mode,
            pilot_symbols=pilots,
            dm_block_valid=dm_valid,
        )

    def _resolve_equalizer(self, has_pilots: bool, has_reference: bool) -> ResolvedEqualizer:
        """The concrete equaliser for :attr:`equalize` (``"auto"`` resolution order in module docs)."""
        if self.equalize != "auto":
            return self.equalize
        if has_pilots:
            return "pilot"
        return "data_aided" if has_reference else "unit_power"

    def _estimate_gain(
        self,
        mode: ResolvedEqualizer,
        raw: npt.NDArray[np.complex128],
        raw_pilots: npt.NDArray[np.complex128] | None,
        pilot_values: npt.NDArray[np.complex128] | None,
        reference: npt.NDArray[np.complex128] | None,
    ) -> npt.NDArray[np.complex128]:
        gain: npt.NDArray[np.complex128]
        if mode == "none":
            return np.ones(raw.shape[:-1], dtype=np.complex128)
        if mode == "unit_power":
            power = np.mean(np.abs(raw) ** 2, axis=-1)
            gain = np.sqrt(power / self.data_symbol_energy).astype(np.complex128)
            return gain
        if mode == "pilot":
            if raw_pilots is None or pilot_values is None or raw_pilots.shape[-1] == 0:
                raise ValueError('equalize="pilot" needs pilots')
            num = np.sum(np.conj(pilot_values) * raw_pilots, axis=-1)
            gain = num / np.sum(np.abs(pilot_values) ** 2, axis=-1)
            return gain
        if mode == "data_aided":
            if reference is None:
                raise ValueError(
                    'equalize="data_aided" needs the transmitted signal: '
                    "receive(field, dt_s, reference=sent)"
                )
            if reference.shape != raw.shape:
                raise ValueError(
                    f"reference data symbols have shape {reference.shape}, received {raw.shape}"
                )
            num = np.sum(np.conj(reference) * raw, axis=-1)
            gain = num / np.sum(np.abs(reference) ** 2, axis=-1)
            return gain
        raise AssertionError(f"unreachable: equalize={mode!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Pulse-shaped receivers
# ---------------------------------------------------------------------------


class PulseShapedReceiver(Receiver, ABC):
    """Common base of :class:`SingleCarrierReceiver` and :class:`~hpcom.wdm.WDMReceiver`.

    The pulse fields (``samples_per_symbol``, ``roll_off``,
    ``pulse_span_symbols``, ``convolution``) and ``pilots`` must equal the
    transmitter's; see :class:`~hpcom.transmitter.PulseShapedTransmitter`.
    """

    samples_per_symbol: int = Field(default=8, ge=2)
    roll_off: float = Field(default=0.1, ge=0.0, le=1.0)
    pulse_span_symbols: int | None = Field(default=None, ge=1)
    convolution: Convolution = "circular"
    pilots: PilotConfig | None = None

    @model_validator(mode="after")
    def _validate_pulse_rx(self) -> PulseShapedReceiver:
        if self.convolution == "linear" and self.pulse_span_symbols is None:
            raise ValueError('convolution="linear" needs a finite `pulse_span_symbols`')
        if self.equalize == "pilot" and self.pilots is None:
            raise ValueError('equalize="pilot" requires `pilots`')
        if self.dm is not None and self.pilots is not None:
            raise ValueError("`pilots` and `dm` cannot be combined")
        return self

    @property
    def transient_symbols(self) -> int:
        return (self.pulse_span_symbols or 0) if self.convolution == "linear" else 0

    def _pulse_issues(self, transmitter: Transmitter) -> list[str]:
        if not isinstance(transmitter, PulseShapedTransmitter):
            return [f"{type(self).__name__} cannot receive a {type(transmitter).__name__}"]
        issues = []
        for name in (
            "samples_per_symbol",
            "roll_off",
            "pulse_span_symbols",
            "convolution",
            "pilots",
        ):
            if getattr(transmitter, name) != getattr(self, name):
                issues.append(
                    f"{name}: transmitter {getattr(transmitter, name)!r}, "
                    f"receiver {getattr(self, name)!r}"
                )
        return issues

    def _demodulate(
        self,
        streams: npt.NDArray[np.complex128],
        reference_data: npt.NDArray[np.complex128] | None,
    ) -> Detection:
        """Matched filter → split data/pilot slots → :meth:`_detect`."""
        slots = matched_filter(
            streams,
            self.samples_per_symbol,
            self.roll_off,
            self.pulse_span_symbols,
            self.convolution,
        )
        n_slots = slots.shape[-1]
        if self.pilots is None:
            return self._detect(slots, reference_data=reference_data)
        p_idx = self.pilots.pilot_indices(n_slots)
        d_idx = self.pilots.data_indices(n_slots)
        pilot_values = np.full(p_idx.size, self.pilots.symbol, dtype=np.complex128)
        return self._detect(
            slots[..., d_idx],
            raw_pilots=slots[..., p_idx],
            pilot_values=pilot_values,
            reference_data=reference_data,
        )


class SingleCarrierReceiver(PulseShapedReceiver):
    """Single-carrier RRC matched-filter receiver (optionally dual-polarisation)."""

    def compatibility_issues(self, transmitter: Transmitter) -> list[str]:
        if not isinstance(transmitter, SingleCarrierTransmitter):
            return [f"SingleCarrierReceiver cannot receive a {type(transmitter).__name__}"]
        return self._base_issues(transmitter) + self._pulse_issues(transmitter)

    def reference(
        self, sent: TransmittedSignal
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]]:
        return np.asarray(sent.bits), np.asarray(sent.data_symbols())

    def receive(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        reference: TransmittedSignal | None = None,
    ) -> SingleCarrierReceivedSymbols:
        x = self._compensate_dispersion(self._check_field(field), dt_s)
        ref = self.reference(reference)[1] if reference is not None else None
        out = self._demodulate(x, ref)
        return SingleCarrierReceivedSymbols(receiver=self, **out)


# Resolve forward references (``ReceivedSymbols.receiver`` refers to ``Receiver``).
ReceivedSymbols.model_rebuild()
SingleCarrierReceivedSymbols.model_rebuild()
