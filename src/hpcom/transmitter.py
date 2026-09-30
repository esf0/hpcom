r"""Optical transmitters: abstract base, pulse-shaped base, single-carrier QAM.

Class hierarchy (the receiver side mirrors it)::

    Transmitter (abstract)                       — modulation, polarisation, power, seed
    ├── PulseShapedTransmitter (abstract)        — symbol rate, RRC pulse, pilots
    │   ├── SingleCarrierTransmitter             — one channel
    │   └── WDMTransmitter  (hpcom.wdm)          — N channels on a frequency grid
    └── OFDMTransmitter     (hpcom.ofdm)         — IFFT/CP multicarrier

Conventions shared by every transmitter:

* **Launch power** is per WDM channel (and per core), summed over both
  polarisations (:attr:`Transmitter.launch_power_per_channel_dbm`, default
  0 dBm). A dual-polarisation channel puts half of it in each polarisation.
  The alternative input :attr:`Transmitter.launch_power_total_dbm` is the sum
  over all channels and cores. The field is returned in physical units (√W), scaled so
  its *expected* average power equals the launch power; one finite
  realisation fluctuates around it by :math:`O(1/\sqrt{n})`.
* **Array layout**: time (or symbol index) on the last axis; leading axes
  are ``(core, polarisation, channel)``, each present only when applicable.
  A single-polarisation single-carrier field is 1-D ``(n_samples,)``; a
  dual-polarisation field is ``(2, n_samples)``; space-division multiplexing
  over ``n_cores`` cores (for a :class:`~hpcom.MulticoreFiber`) adds a
  leading core axis, ``(n_cores, 2, n_samples)`` — the multicore field
  layout.
* **bits** are information bits — the input of the bit→symbol mapping (or of
  the distribution matcher). Pilots carry no bits.
* **symbols** are on the unit-power constellation scale (not scaled by the
  launch power) and include pilot slots.
* **Randomness**: stream ``i`` (in ``(core, polarisation, channel)`` C order)
  draws from child ``i`` of ``SeedSequence(seed)`` (or of a seed drawn from
  the ``rng`` argument), so streams are independent and reproducible.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import ComplexArrayView, HpcomModel, Polymorphic, UInt8ArrayView
from hpcom._pulse import Convolution, pulse_shape
from hpcom._typing import ComplexArray, Polarization
from hpcom.modulation.spec import ModulationSpec
from hpcom.pilots import PilotConfig
from hpcom.units import dbm_to_w

if TYPE_CHECKING:
    from hpcom.receiver import Receiver, SingleCarrierReceiver

_Tx = TypeVar("_Tx", bound="Transmitter")

__all__ = [
    "Polarization",
    "PulseShapedTransmitter",
    "SingleCarrierSignal",
    "SingleCarrierTransmitter",
    "TransmittedSignal",
    "Transmitter",
]


# ---------------------------------------------------------------------------
# Transmitter base
# ---------------------------------------------------------------------------


class Transmitter(ModulationSpec, ABC):
    """Abstract optical transmitter.

    Attributes
    ----------
    polarization
        ``"single"`` or ``"dual"`` (polarisation-division multiplexing; the two
        tributaries carry independent data).
    launch_power_per_channel_dbm
        Average launch power **per WDM channel (per core), summed over
        polarisations**, in dBm. Default 0 dBm when neither power field is
        given.
    launch_power_total_dbm
        Alternative to the above: total launch power of all channels, cores
        and polarisations. Set at most one of the two; the other is derived
        (``P_total = n_cores · n_channels · P_channel``).
    n_cores
        Space-division multiplexing: independent data on each of
        ``n_cores`` cores (fields ``(n_cores, …, n_samples)``), for links of
        :class:`~hpcom.MulticoreFiber`.
    seed
        Seed for the random data. ``None`` draws fresh entropy (unless an
        ``rng`` is passed to :meth:`generate`).
    """

    polarization: Polarization = "single"
    launch_power_per_channel_dbm: float | None = None
    launch_power_total_dbm: float | None = None
    n_cores: int = Field(default=1, ge=1)
    seed: int | None = None

    _exclusive_fields: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("modulation_order", "constellation"),
        ("launch_power_per_channel_dbm", "launch_power_total_dbm"),
    )

    @model_validator(mode="after")
    def _validate_transmitter(self) -> Transmitter:
        if (
            self.launch_power_per_channel_dbm is not None
            and self.launch_power_total_dbm is not None
        ):
            raise ValueError(
                "set at most one of `launch_power_per_channel_dbm` and `launch_power_total_dbm`"
            )
        n_data = self.n_data_symbols_per_stream
        if n_data < 1:
            raise ValueError("the signal has no data symbols")
        if self.dm is not None and n_data % self.dm.block_length:
            raise ValueError(
                f"data symbols per stream ({n_data}) must be a multiple of "
                f"dm.block_length ({self.dm.block_length})"
            )
        return self

    # ---- structure ---------------------------------------------------------

    @property
    def n_polarizations(self) -> int:
        return 2 if self.polarization == "dual" else 1

    @property
    def n_channels(self) -> int:
        """Number of WDM channels (1 unless overridden)."""
        return 1

    @property
    @abstractmethod
    def dt_s(self) -> float:
        """Sample period of the generated field, in seconds."""

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.dt_s

    @property
    @abstractmethod
    def n_data_symbols_per_stream(self) -> int:
        """Data (information-carrying) symbols per polarisation per channel."""

    @property
    def n_bits_per_stream(self) -> int:
        """Information bits per polarisation per channel."""
        n = self.n_data_symbols_per_stream
        if self.dm is not None:
            return (n // self.dm.block_length) * self.dm.bits_per_block
        return n * self.bits_per_symbol

    @property
    def stream_shape(self) -> tuple[int, ...]:
        """Leading shape of ``bits`` / ``symbols``: ``(n_cores?, 2?, n_channels?)``."""
        shape: tuple[int, ...] = (self.n_cores,) if self.n_cores > 1 else ()
        if self.polarization == "dual":
            shape += (2,)
        if self.n_channels > 1:
            shape += (self.n_channels,)
        return shape

    # ---- launch power ------------------------------------------------------

    @property
    def launch_power_per_channel_w(self) -> float:
        """Average power per channel and core (both polarisations), in W."""
        if self.launch_power_total_dbm is not None:
            return dbm_to_w(self.launch_power_total_dbm) / (self.n_channels * self.n_cores)
        return dbm_to_w(self.launch_power_per_channel_dbm or 0.0)

    @property
    def launch_power_total_w(self) -> float:
        """Average power of all channels, cores and polarisations, in W."""
        return self.launch_power_per_channel_w * self.n_channels * self.n_cores

    @property
    def launch_power_per_polarization_w(self) -> float:
        """Average power per channel per polarisation, in W."""
        return self.launch_power_per_channel_w / self.n_polarizations

    def with_launch_power(
        self: _Tx, *, per_channel_dbm: float | None = None, total_dbm: float | None = None
    ) -> _Tx:
        """Copy with a new launch power (exactly one argument)."""
        if (per_channel_dbm is None) == (total_dbm is None):
            raise TypeError("pass exactly one of per_channel_dbm= or total_dbm=")
        if per_channel_dbm is not None:
            return self.replace(launch_power_per_channel_dbm=per_channel_dbm)
        return self.replace(launch_power_total_dbm=total_dbm)

    def matched_receiver(self, **overrides: Any) -> Receiver:
        """A receiver for this transmitter's signal.

        Modulation, distribution matcher, polarisation and the waveform's own
        settings (pulse, pilots, WDM grid, OFDM frame, …) are copied;
        ``overrides`` set anything else — typically ``equalize`` and
        ``cd_compensation_s2`` — or replace a copied value.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define matched_receiver(); construct its receiver directly"
        )

    def _modulation_fields(self) -> dict[str, Any]:
        return {
            "modulation_order": self.modulation_order,
            "constellation": self.constellation,
            "dm": self.dm,
            "polarization": self.polarization,
            "n_cores": self.n_cores,
        }

    # ---- generation --------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.ArrayLike | None = None,
        symbols: npt.ArrayLike | None = None,
    ) -> TransmittedSignal:
        """Generate a signal.

        With neither ``bits`` nor ``symbols``, data is random (from ``rng`` if
        given, else from :attr:`seed`). For deterministic input pass exactly
        one of:

        * ``bits`` — shape ``(*stream_shape, n_bits_per_stream)``;
        * ``symbols`` — data symbols on the unit-power constellation scale,
          shape ``(*stream_shape, n_data_symbols_per_stream)``. They are
          transmitted as given (need not be constellation points); ``bits``
          are the demapped labels of their nearest points (DM-decoded when a
          DM is set).
        """

    def _draw_data(
        self,
        rng: np.random.Generator | None,
        bits: npt.ArrayLike | None,
        symbols: npt.ArrayLike | None,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]]:
        """Information bits and data symbols for every stream.

        Returns arrays of shape ``(*stream_shape, n_bits_per_stream)`` and
        ``(*stream_shape, n_data_symbols_per_stream)``.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`")
        lead = self.stream_shape
        n_bits, n_data = self.n_bits_per_stream, self.n_data_symbols_per_stream
        cfg = self.resolved_constellation
        dm = self.dm

        if symbols is not None:
            syms = np.asarray(symbols, dtype=np.complex128)
            if syms.shape != (*lead, n_data):
                raise ValueError(f"`symbols` must have shape {(*lead, n_data)}, got {syms.shape}")
            flat_syms = syms.reshape(-1, n_data)
            idx = cfg.nearest(flat_syms)
            if dm is not None:
                out_bits = np.stack([dm.decode(row) for row in idx])
            else:
                out_bits = cfg.bit_labels[idx].reshape(flat_syms.shape[0], -1)
            return out_bits.reshape(*lead, n_bits), syms.copy()

        if bits is not None:
            b = np.asarray(bits)
            if b.shape != (*lead, n_bits):
                raise ValueError(f"`bits` must have shape {(*lead, n_bits)}, got {b.shape}")
            if np.any((b != 0) & (b != 1)):
                raise ValueError("`bits` must contain only 0 and 1")
            flat_bits = b.astype(np.uint8).reshape(-1, n_bits)
            idx = np.stack(
                [
                    dm.encode(row) if dm is not None else cfg.indices_from_bits(row)
                    for row in flat_bits
                ]
            )
            return flat_bits.reshape(*lead, n_bits).copy(), cfg.points[idx].reshape(*lead, n_data)

        root = _seed_sequence(rng, self.seed)
        n_streams = math.prod(lead)
        out_bits = np.empty((n_streams, n_bits), dtype=np.uint8)
        out_syms = np.empty((n_streams, n_data), dtype=np.complex128)
        for i, child in enumerate(root.spawn(n_streams)):
            gen = np.random.default_rng(child)
            if dm is None and cfg.probabilities is not None:
                # i.i.d. probabilistic shaping: draw symbols from the PMF.
                idx = gen.choice(cfg.modulation_order, size=n_data, p=cfg.probabilities)
                out_bits[i] = cfg.bits_from_indices(idx)
            else:
                out_bits[i] = gen.integers(0, 2, size=n_bits, dtype=np.uint8)
                idx = (
                    dm.encode(out_bits[i]) if dm is not None else cfg.indices_from_bits(out_bits[i])
                )
            out_syms[i] = cfg.points[idx]
        return out_bits.reshape(*lead, n_bits), out_syms.reshape(*lead, n_data)


def _seed_sequence(rng: np.random.Generator | None, seed: int | None) -> np.random.SeedSequence:
    if rng is not None:
        return np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
    return np.random.SeedSequence(seed)


# ---------------------------------------------------------------------------
# Transmitted signal
# ---------------------------------------------------------------------------


class TransmittedSignal(HpcomModel, ABC):
    """Output of :meth:`Transmitter.generate` (abstract; one subclass per waveform).

    Attributes
    ----------
    transmitter
        The configuration that produced this signal.
    field
        Complex envelope in √W: ``(n_samples,)`` or ``(2, n_samples)``.
    symbols
        Transmitted symbol stream on the unit-power constellation scale,
        ``(*transmitter.stream_shape, n_slots)``, including pilot slots.
    bits
        Information bits, ``(*transmitter.stream_shape, n_bits_per_stream)``.
    """

    transmitter: Polymorphic[Transmitter]
    field: ComplexArrayView
    symbols: ComplexArrayView
    bits: UInt8ArrayView

    @property
    def dt_s(self) -> float:
        return self.transmitter.dt_s

    @property
    def polarization(self) -> Polarization:
        return self.transmitter.polarization

    @property
    @abstractmethod
    def data_indices(self) -> npt.NDArray[np.intp]:
        """Positions of data symbols along the last axis of :attr:`symbols`."""

    def data_symbols(self) -> ComplexArray:
        """Data symbols only, ``(*stream_shape, n_data_symbols_per_stream)``."""
        out: ComplexArray = self.symbols[..., self.data_indices]
        return out

    @property
    def mean_power_w(self) -> float:
        """Realised mean power of :attr:`field` summed over polarisations, in W."""
        p = np.mean(np.abs(self.field) ** 2, axis=-1)
        return float(np.sum(p))


# ---------------------------------------------------------------------------
# Pulse-shaped transmitters (single carrier, WDM)
# ---------------------------------------------------------------------------


class PulseShapedTransmitter(Transmitter, ABC):
    """Common base of :class:`SingleCarrierTransmitter` and :class:`~hpcom.wdm.WDMTransmitter`.

    Each stream is a sequence of ``n_symbols`` slots at ``symbol_rate_hz``,
    shaped by a root-raised-cosine pulse and sampled ``samples_per_symbol``
    times per symbol.

    Attributes
    ----------
    symbol_rate_hz
        Symbol rate ``R_s`` in baud.
    n_symbols
        Symbol slots per stream, pilots included.
    samples_per_symbol
        Oversampling factor (``dt = 1/(R_s·sps)``).
    roll_off
        RRC roll-off β ∈ [0, 1].
    pulse_span_symbols
        ``None`` → exact RRC (frequency-domain, periodic). An integer ``S`` →
        truncated FIR of ``S·sps + 1`` taps (a realistic DSP filter).
    convolution
        ``"circular"`` → the signal is periodic (consistent with FFT-based
        fiber propagation, no block edges). ``"linear"`` → a finite burst;
        the ≈``pulse_span_symbols`` symbols at each end carry filter
        transients (requires a finite span).
    pilots
        Optional :class:`~hpcom.pilots.PilotConfig` (time-domain pilots).
    """

    symbol_rate_hz: float = Field(gt=0)
    n_symbols: int = Field(ge=1)
    samples_per_symbol: int = Field(default=8, ge=2)
    roll_off: float = Field(default=0.1, ge=0.0, le=1.0)
    pulse_span_symbols: int | None = Field(default=None, ge=1)
    convolution: Convolution = "circular"
    pilots: PilotConfig | None = None

    @model_validator(mode="after")
    def _validate_pulse(self) -> PulseShapedTransmitter:
        if self.convolution == "linear" and self.pulse_span_symbols is None:
            raise ValueError('convolution="linear" needs a finite `pulse_span_symbols`')
        if self.pulse_span_symbols is not None and self.pulse_span_symbols >= self.n_symbols:
            raise ValueError(
                f"pulse_span_symbols ({self.pulse_span_symbols}) must be smaller than "
                f"n_symbols ({self.n_symbols})"
            )
        if self.dm is not None and self.pilots is not None:
            raise ValueError("`pilots` and `dm` cannot be combined")
        return self

    @property
    def dt_s(self) -> float:
        return 1.0 / (self.symbol_rate_hz * self.samples_per_symbol)

    @property
    def n_samples(self) -> int:
        return self.n_symbols * self.samples_per_symbol

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        """Data slots within ``n_symbols``."""
        if self.pilots is None:
            return np.arange(self.n_symbols, dtype=np.intp)
        return self.pilots.data_indices(self.n_symbols)

    @property
    def pilot_indices(self) -> npt.NDArray[np.intp]:
        """Pilot slots within ``n_symbols`` (empty without pilots)."""
        if self.pilots is None:
            return np.empty(0, dtype=np.intp)
        return self.pilots.pilot_indices(self.n_symbols)

    @property
    def n_data_symbols_per_stream(self) -> int:
        return int(self.data_indices.size)

    @property
    def transient_symbols(self) -> int:
        """Symbols at each block edge affected by filter transients (0 when circular)."""
        return (self.pulse_span_symbols or 0) if self.convolution == "linear" else 0

    def _pulse_fields(self) -> dict[str, Any]:
        return {
            **self._modulation_fields(),
            "samples_per_symbol": self.samples_per_symbol,
            "roll_off": self.roll_off,
            "pulse_span_symbols": self.pulse_span_symbols,
            "convolution": self.convolution,
            "pilots": self.pilots,
        }

    def _slot_stream(self, data: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """Insert pilots: ``(..., n_data)`` → ``(..., n_symbols)``."""
        if self.pilots is None:
            return data
        stream = np.empty((*data.shape[:-1], self.n_symbols), dtype=np.complex128)
        stream[..., self.pilot_indices] = self.pilots.symbol
        stream[..., self.data_indices] = data
        return stream

    def _stream_energy(self) -> float:
        """Expected ``|s|²`` per slot, pilots included."""
        n_pil = self.pilot_indices.size
        pilot_energy = abs(self.pilots.symbol) ** 2 if self.pilots is not None else 0.0
        e_data = self.data_symbol_energy
        return (self.n_data_symbols_per_stream * e_data + n_pil * pilot_energy) / self.n_symbols

    def _shape_streams(self, slots: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """Pulse-shape every stream and scale to the per-polarisation launch power."""
        # > 0: constellations have unit energy and every stream has a data symbol.
        scale = np.sqrt(self.launch_power_per_polarization_w / self._stream_energy())
        shaped = pulse_shape(
            slots, self.samples_per_symbol, self.roll_off, self.pulse_span_symbols, self.convolution
        )
        out: npt.NDArray[np.complex128] = shaped * scale
        return out


class SingleCarrierSignal(TransmittedSignal):
    """Output of :class:`SingleCarrierTransmitter`.

    Shapes: single-pol ``field (n_samples,)``, ``symbols (n_symbols,)``,
    ``bits (n_bits,)``; dual-pol adds a leading axis of 2.
    """

    transmitter: Polymorphic[SingleCarrierTransmitter]

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        return self.transmitter.data_indices

    @property
    def pilot_indices(self) -> npt.NDArray[np.intp]:
        return self.transmitter.pilot_indices


class SingleCarrierTransmitter(PulseShapedTransmitter):
    """Single-carrier RRC-shaped QAM transmitter (optionally dual-polarisation).

    >>> tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9,
    ...                               n_symbols=1024, seed=0)
    >>> sig = tx.generate()
    >>> sig.field.shape, sig.bits.shape
    ((8192,), (4096,))
    """

    def matched_receiver(self, **overrides: Any) -> SingleCarrierReceiver:
        from hpcom.receiver import SingleCarrierReceiver

        return SingleCarrierReceiver(**{**self._pulse_fields(), **overrides})

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.ArrayLike | None = None,
        symbols: npt.ArrayLike | None = None,
    ) -> SingleCarrierSignal:
        out_bits, data = self._draw_data(rng, bits, symbols)
        slots = self._slot_stream(data)
        return SingleCarrierSignal(
            transmitter=self, field=self._shape_streams(slots), symbols=slots, bits=out_bits
        )


# Resolve forward references (signal classes refer to transmitters defined after them).
SingleCarrierSignal.model_rebuild()
