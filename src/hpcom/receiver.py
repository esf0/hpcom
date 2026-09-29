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

   *Or*, for pulse-shaped and OFDM receivers, a **compensation stage**
   (``compensation=``, e.g. :class:`~hpcom.dsp.DigitalBackPropagation`) run
   on realistic front-end samples: per channel shifted to baseband (OFDM:
   the whole band), band-limited and resampled to the stage's rate (see
   :class:`~hpcom.dsp.Compensator`).
2. **Waveform-specific demodulation** to symbol-rate samples (matched
   filter — at the compensation stage's rate if one is set — or FFT for
   OFDM).
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
   ``"mimo"`` (pulse-shaped receivers)
       A fractionally spaced (2 samples/symbol) butterfly FIR of
       ``mimo_taps`` taps — 2×2 for dual polarisation, per channel —
       fitted by least squares to the transmitted data symbols (the genie
       of ``"data_aided"``; needs ``reference=``). Pilots are equalised
       too but do not train it: one constant pilot symbol cannot identify
       a multi-tap filter, nor tell X from Y.
       :math:`y_p[k] = \sum_{q}\sum_{j} w_{pq,j}\,x_q[2k+j]`. Undoes
       polarisation rotation, PMD and residual linear ISI; it is the
       least-squares limit of the adaptive (CMA/LMS) equalisers of real
       receivers.
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
from hpcom._pulse import Convolution, matched_filter, matched_filter_samples
from hpcom._typing import ComplexArray, Polarization
from hpcom.dsp.compensation import Compensator
from hpcom.dsp.frontend import lowpass, resample
from hpcom.modulation.spec import ModulationSpec
from hpcom.pilots import PilotConfig
from hpcom.transmitter import (
    PulseShapedTransmitter,
    SingleCarrierTransmitter,
    TransmittedSignal,
    Transmitter,
)

#: An equaliser that is actually applied (``"auto"`` resolves to one of these).
ResolvedEqualizer = Literal["none", "unit_power", "pilot", "data_aided", "mimo"]
#: Equaliser selection (see the module docstring).
EqualizerMode = Literal["auto", "none", "unit_power", "pilot", "data_aided", "mimo"]


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
# Compensation stage (shared by pulse-shaped and OFDM receivers)
# ---------------------------------------------------------------------------


def _check_compensation_exclusive(receiver: Receiver, compensation: Compensator | None) -> None:
    if compensation is not None and (
        receiver.cd_compensation_s2 != 0.0 or receiver.cd_compensation_s3 != 0.0
    ):
        raise ValueError(
            "`compensation` already compensates dispersion; "
            "leave cd_compensation_s2 and cd_compensation_s3 at 0"
        )


def _is_identity_front_end(compensation: Compensator | None, samples_per_symbol: int) -> bool:
    """The front end changes nothing: the stage runs once on the whole field."""
    return (
        compensation is not None
        and compensation.bandwidth_hz is None
        and compensation.samples_per_symbol in (None, samples_per_symbol)
    )


def _compensation_front_end(
    streams: npt.NDArray[np.complex128],
    dt_s: float,
    samples_per_symbol: int,
    compensation: Compensator,
) -> tuple[npt.NDArray[np.complex128], float, int]:
    """Low-pass to ``bandwidth_hz`` and resample to the stage's samples per symbol.

    ``samples_per_symbol`` is the input rate over the (aggregate) symbol rate:
    the pulse-shaped ``samples_per_symbol``, or the OFDM oversampling ``L``
    (symbol rate ``N·Δf``). Returns ``(samples, dt_s, samples_per_symbol)``.
    """
    sps_in, n = samples_per_symbol, streams.shape[-1]
    if n % sps_in:
        raise ValueError(f"field length {n} is not a multiple of samples_per_symbol={sps_in}")
    sps_out = compensation.samples_per_symbol or sps_in
    symbol_rate_hz = 1.0 / (dt_s * sps_in)
    rate_hz = sps_out * symbol_rate_hz
    if compensation.bandwidth_hz is not None:
        if compensation.bandwidth_hz > rate_hz * (1.0 + 1e-12):
            raise ValueError(
                f"compensation.bandwidth_hz = {compensation.bandwidth_hz:.4g} Hz exceeds the "
                f"front-end sampling rate {rate_hz:.4g} Hz ({sps_out} samples/symbol "
                f"at {symbol_rate_hz:.4g} Bd); raise samples_per_symbol"
            )
        streams = lowpass(streams, dt_s, compensation.bandwidth_hz)
    if sps_out != sps_in:
        streams = resample(streams, n // sps_in * sps_out)
    return streams, 1.0 / rate_hz, sps_out


def _checked(
    compensation: Compensator, out: ComplexArray, shape: tuple[int, ...]
) -> npt.NDArray[np.complex128]:
    arr = np.asarray(out, dtype=np.complex128)
    if arr.shape != shape:
        raise ValueError(
            f"{type(compensation).__name__}.compensate returned shape {arr.shape}, expected {shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# Pulse-shaped receivers
# ---------------------------------------------------------------------------


class PulseShapedReceiver(Receiver, ABC):
    """Common base of :class:`SingleCarrierReceiver` and :class:`~hpcom.wdm.WDMReceiver`.

    The pulse fields (``samples_per_symbol``, ``roll_off``,
    ``pulse_span_symbols``, ``convolution``) and ``pilots`` must equal the
    transmitter's; see :class:`~hpcom.transmitter.PulseShapedTransmitter`.

    ``compensation`` (optional) is a :class:`~hpcom.dsp.Compensator` — e.g.
    :class:`~hpcom.dsp.DigitalBackPropagation` — run per channel on the
    samples returned by :meth:`front_end`, in place of
    ``cd_compensation_s2``/``_s3`` (which must then be 0). The matched filter
    then runs at the stage's ``samples_per_symbol``.

    ``mimo_taps`` (odd) is the length of the ``equalize="mimo"`` FIR, in
    samples at 2 samples/symbol (1 when the matched filter runs at 1).
    """

    samples_per_symbol: int = Field(default=8, ge=2)
    roll_off: float = Field(default=0.1, ge=0.0, le=1.0)
    pulse_span_symbols: int | None = Field(default=None, ge=1)
    convolution: Convolution = "circular"
    pilots: PilotConfig | None = None
    compensation: Polymorphic[Compensator] | None = None
    mimo_taps: int = Field(default=15, ge=1)

    @model_validator(mode="after")
    def _validate_pulse_rx(self) -> PulseShapedReceiver:
        _check_compensation_exclusive(self, self.compensation)
        if self.mimo_taps % 2 == 0:
            raise ValueError(f"mimo_taps must be odd, got {self.mimo_taps}")
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

    # ---- front end and compensation ----------------------------------------

    def front_end(
        self, field: ComplexArray, dt_s: float
    ) -> tuple[npt.NDArray[np.complex128], float]:
        """The samples the compensation stage receives, and their sample period.

        Every selected channel shifted to baseband, low-passed to
        ``compensation.bandwidth_hz`` and resampled to
        ``compensation.samples_per_symbol`` (steps 1–3 of
        :class:`~hpcom.dsp.Compensator`): shape ``(*pol, n)`` for a single
        carrier, ``(*pol, n_selected, n)`` for WDM. Without ``compensation``,
        the baseband channels at the simulation rate.

        When the front end changes nothing (``samples_per_symbol`` ``None``
        or equal to the receiver's, ``bandwidth_hz`` ``None``) the stage
        processes the **whole received field** once, centred at 0 Hz —
        full-field processing (for WDM, full-field DBP) — and this returns
        that field, ``(*pol, n)``.

        Use it to feed a custom algorithm, or to build datasets, from exactly
        what the receiver's compensation stage sees.
        """
        x = self._check_field(field)
        if self._full_field_compensation:
            return x.copy(), dt_s
        return self._front_end(self._baseband(x, dt_s), dt_s)

    @property
    def _full_field_compensation(self) -> bool:
        """``True`` if ``compensation`` runs once on the whole field (identity front end)."""
        return _is_identity_front_end(self.compensation, self.samples_per_symbol)

    def _baseband(
        self, field: npt.NDArray[np.complex128], dt_s: float
    ) -> npt.NDArray[np.complex128]:
        """Shift the channel(s) to 0 Hz (identity for a single carrier)."""
        del dt_s
        return field

    def _channel_offsets_hz(self) -> tuple[float, ...] | None:
        """Baseband frequency of each channel along axis −2 (``None``: no channel axis, 0 Hz)."""
        return None

    def _front_end(
        self, streams: npt.NDArray[np.complex128], dt_s: float
    ) -> tuple[npt.NDArray[np.complex128], float]:
        comp = self.compensation
        if comp is None:
            return streams, dt_s
        samples, dt_out, _ = _compensation_front_end(streams, dt_s, self.samples_per_symbol, comp)
        return samples, dt_out

    def _receive_field(
        self,
        field: npt.NDArray[np.complex128],
        dt_s: float,
        reference_data: npt.NDArray[np.complex128] | None,
    ) -> Detection:
        """Field → baseband → front end → compensation → matched filter → detection."""
        comp = self.compensation
        if comp is not None and self._full_field_compensation:
            field = self._checked(comp.compensate(field, dt_s), field.shape)
            comp = None
        streams = self._baseband(field, dt_s)
        if comp is None:
            return self._demodulate(streams, reference_data, self.samples_per_symbol)
        samples, dt_c = self._front_end(streams, dt_s)
        offsets = self._channel_offsets_hz()
        if offsets is None:
            compensated = self._checked(comp.compensate(samples, dt_c), samples.shape)
        else:
            compensated = np.empty_like(samples)
            for c, f_hz in enumerate(offsets):
                one = samples[..., c, :]
                compensated[..., c, :] = self._checked(
                    comp.compensate(one, dt_c, frequency_offset_hz=f_hz), one.shape
                )
        return self._demodulate(
            compensated, reference_data, comp.samples_per_symbol or self.samples_per_symbol
        )

    def _checked(self, out: ComplexArray, shape: tuple[int, ...]) -> npt.NDArray[np.complex128]:
        assert self.compensation is not None
        return _checked(self.compensation, out, shape)

    def _demodulate(
        self,
        streams: npt.NDArray[np.complex128],
        reference_data: npt.NDArray[np.complex128] | None,
        samples_per_symbol: int,
    ) -> Detection:
        """Matched filter at ``samples_per_symbol`` → split data/pilot slots → :meth:`_detect`."""
        if self.equalize == "mimo":
            return self._demodulate_mimo(streams, reference_data, samples_per_symbol)
        slots = matched_filter(
            streams,
            samples_per_symbol,
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

    def _demodulate_mimo(
        self,
        streams: npt.NDArray[np.complex128],
        reference_data: npt.NDArray[np.complex128] | None,
        samples_per_symbol: int,
    ) -> Detection:
        """Matched filter → 2 samples/symbol → least-squares butterfly FIR → :meth:`_detect`."""
        filtered = matched_filter_samples(
            streams, samples_per_symbol, self.roll_off, self.pulse_span_symbols, self.convolution
        )
        n_slots = filtered.shape[-1] // samples_per_symbol
        if samples_per_symbol == 1:
            per = 1
        elif samples_per_symbol % 2 == 0:
            per, filtered = 2, filtered[..., :: samples_per_symbol // 2]
        else:
            per, filtered = 2, resample(filtered, 2 * n_slots)
        if reference_data is None:
            raise ValueError(
                'equalize="mimo" is trained on the transmitted data: '
                "receive(field, dt_s, reference=sent)"
            )
        train = np.arange(n_slots) if self.pilots is None else self.pilots.data_indices(n_slots)
        targets = np.asarray(reference_data, dtype=np.complex128)
        slots = mimo_equalize(
            filtered, per, self.mimo_taps, self.polarization == "dual", train, targets
        )
        gain = np.ones(slots.shape[:-1], dtype=np.complex128)
        if self.pilots is None:
            return self._detect(slots, gain=gain, equalizer="mimo", reference_data=reference_data)
        d_idx = self.pilots.data_indices(n_slots)
        p_idx = self.pilots.pilot_indices(n_slots)
        return self._detect(
            slots[..., d_idx],
            raw_pilots=slots[..., p_idx],
            pilot_values=np.full(p_idx.size, self.pilots.symbol, dtype=np.complex128),
            gain=gain,
            equalizer="mimo",
        )


def mimo_equalize(
    samples: npt.NDArray[np.complex128],
    samples_per_symbol: int,
    n_taps: int,
    dual: bool,
    train_slots: npt.NDArray[np.intp],
    targets: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    r"""Least-squares butterfly FIR equaliser (see ``equalize="mimo"``).

    ``samples``: ``(*pol, *channels, S·n)`` at ``S`` = ``samples_per_symbol``
    (1 or 2); ``targets``: ``(*pol, *channels, len(train_slots))``. For each
    channel, the taps :math:`w` minimise :math:`\sum_{k\in\text{train}}
    |t_p[k] - \sum_{q,j} w_{pq,j}\,x_q[Sk + j]|^2` (circular indices,
    :math:`j = -(T-1)/2 … (T-1)/2`); returns the equalised symbols
    ``(*pol, *channels, n)``.
    """
    x = np.asarray(samples, dtype=np.complex128)
    t = np.asarray(targets, dtype=np.complex128)
    if dual:  # (2, *ch, N) → (*ch, 2, N)
        x, t = np.moveaxis(x, 0, -2), np.moveaxis(t, 0, -2)
    else:
        x, t = x[..., None, :], t[..., None, :]
    lead = x.shape[:-2]
    n_pol, n_samples = x.shape[-2], x.shape[-1]
    n_slots = n_samples // samples_per_symbol
    half = (n_taps - 1) // 2
    index = (
        samples_per_symbol * np.arange(n_slots)[:, None] + np.arange(-half, half + 1)
    ) % n_samples
    flat_x = x.reshape(-1, n_pol, n_samples)
    flat_t = t.reshape(-1, n_pol, t.shape[-1])
    out = np.empty((flat_x.shape[0], n_pol, n_slots), dtype=np.complex128)
    for c in range(flat_x.shape[0]):
        regressors = flat_x[c][:, index].transpose(1, 0, 2).reshape(n_slots, n_pol * n_taps)
        taps, *_ = np.linalg.lstsq(regressors[train_slots], flat_t[c].T, rcond=None)
        out[c] = (regressors @ taps).T
    y = out.reshape(*lead, n_pol, n_slots)
    return np.moveaxis(y, -2, 0) if dual else y[..., 0, :]


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
        out = self._receive_field(x, dt_s, ref)
        return SingleCarrierReceivedSymbols(receiver=self, **out)


# Resolve forward references (``ReceivedSymbols.receiver`` refers to ``Receiver``).
ReceivedSymbols.model_rebuild()
SingleCarrierReceivedSymbols.model_rebuild()
