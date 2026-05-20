"""Optical receiver hierarchy: matched filter + downsample + symbol demap.

Mirrors :mod:`hpcom.transmitter`:

* :class:`Receiver` and :class:`ReceivedSymbols` are abstract bases that fix
  the cross-carrier vocabulary (modulation order, polarisation, pilots) — they
  carry no signal-shape-specific machinery.
* :class:`SingleCarrierReceiver` is the concrete RRC matched-filter
  implementation; :class:`SingleCarrierReceivedSymbols` is its output.
* :class:`hpcom.wdm.WDMReceiver` extends :class:`SingleCarrierReceiver` to
  downshift the selected channel before delegating to the single-carrier
  pipeline.
* :class:`hpcom.ofdm.OFDMReceiver` (deferred) will subclass :class:`Receiver`
  directly — no RRC.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import (
    ConstantCompositionDM,
    ConstellationConfig,
    bits_per_symbol,
    nearest_symbol_indices,
)
from hpcom.pilots import PilotConfig
from hpcom.transmitter import Polarization

#: How (or whether) to rescale received-symbol amplitudes before hard decision.
EqualizerMode = Literal["none", "unit_power"]


class ReceivedSymbols(BaseModel):
    """Abstract output of :meth:`Receiver.receive` — concrete subclasses live
    next to their receiver implementations."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    decisions: npt.NDArray[np.intp]
    modulation_order: int
    polarization: Polarization = "single"
    pilots: PilotConfig | None = None
    pilot_symbols: ComplexArray | None = None

    def data_bits(self) -> npt.NDArray[np.uint8]:
        """Bits at data positions only (Option α — pilots stripped).

        Preserves the leading dims of :attr:`bits` (e.g. polarisation axis).
        """
        if self.pilots is None:
            return self.bits
        k = bits_per_symbol(self.modulation_order)
        n_symbols = self.symbols.shape[-1]
        data_idx = self.pilots.data_indices(n_symbols)
        if self.bits.ndim == 1:
            return self.bits.reshape(-1, k)[data_idx].ravel()
        per_stream = self.bits.reshape(self.bits.shape[0], -1, k)
        return per_stream[:, data_idx, :].reshape(self.bits.shape[0], -1)

    def data_symbols(self) -> ComplexArray:
        if self.pilots is None:
            return self.symbols
        n_symbols = self.symbols.shape[-1]
        return self.symbols[..., self.pilots.data_indices(n_symbols)]


class SingleCarrierReceivedSymbols(ReceivedSymbols):
    """Output of :class:`SingleCarrierReceiver` — no extra fields, distinct
    type for downstream isinstance checks."""


class Receiver(BaseModel):
    """Abstract receiver base — defines the cross-carrier vocabulary every
    concrete receiver must expose.

    Concrete subclasses (:class:`SingleCarrierReceiver`,
    :class:`hpcom.wdm.WDMReceiver`, :class:`hpcom.ofdm.OFDMReceiver`) implement
    :meth:`receive` for their specific signal shape.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    modulation_order: int | None = None
    constellation: ConstellationConfig | None = None
    dm: ConstantCompositionDM | None = None
    samples_per_symbol: int = 16
    polarization: Polarization = "single"
    pilots: PilotConfig | None = None

    @field_validator("polarization", mode="before")
    @classmethod
    def _check_polarization(cls, v: object) -> object:
        if v not in ("single", "dual"):
            raise ValueError(
                f"polarization must be 'single' or 'dual', got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _validate_base(self) -> Receiver:
        if (self.modulation_order is None) == (self.constellation is None):
            raise ValueError(
                "exactly one of `modulation_order` or `constellation` must be set"
            )
        if self.modulation_order is not None:
            resolved = ConstellationConfig.from_square_qam(self.modulation_order)
        else:
            assert self.constellation is not None
            resolved = self.constellation
        # Cache resolved constellation on the instance — bypass frozen.
        self.__dict__["_constellation"] = resolved

        if self.dm is not None:
            if self.pilots is not None:
                raise ValueError(
                    "`pilots` and `dm` are mutually exclusive on the Receiver"
                )
            if self.constellation is None or self.constellation.probabilities is None:
                raise ValueError(
                    "`dm` requires `constellation` with non-None probabilities"
                )
            if self.dm.target_pmf.shape != self.constellation.probabilities.shape or not (
                np.allclose(self.dm.target_pmf, self.constellation.probabilities, atol=1e-12)
            ):
                raise ValueError(
                    "dm.target_pmf must match constellation.probabilities elementwise"
                )
        return self

    @property
    def effective_modulation_order(self) -> int:
        """Constellation size — always defined, even when ``modulation_order`` is None."""
        return self._resolved_constellation().modulation_order

    def _resolved_constellation(self) -> ConstellationConfig:
        return self.__dict__["_constellation"]  # type: ignore[no-any-return]

    @abstractmethod
    def receive(
        self, field: ComplexArray, dt_s: float | None = None
    ) -> ReceivedSymbols:
        """Demodulate a received field. Subclasses define the pipeline."""
        raise NotImplementedError


class SingleCarrierReceiver(Receiver):
    """Single-carrier QAM receiver (optionally dual-polarisation).

    Applies the matched RRC filter, downsamples at the symbol rate, optionally
    equalises amplitude, then hard-decodes to nearest constellation point.

    ``equalize`` is explicit: ``"unit_power"`` rescales post-downsample symbols
    so their measured average power equals 1 (handy for noiseless benchmarks
    but masks gain/loss errors); ``"none"`` (default) surfaces gain/loss as BER
    (required for any meaningful AWGN BER curve).
    """

    roll_off: float = 0.1
    pulse_span_symbols: int = 64
    equalize: EqualizerMode = "none"
    cd_compensation_s2: float = 0.0
    """Accumulated dispersion ``β₂ · L_total`` (in s²) to undo before the
    matched filter. Set to ``fiber.beta2_s2_per_m * fiber.length_km * 1e3 * n_spans``
    to remove the chromatic dispersion of an EDFA-compensated link. Leave at
    ``0.0`` for back-to-back / dispersion-free scenarios."""

    @field_validator("equalize", mode="before")
    @classmethod
    def _check_equalize(cls, v: object) -> object:
        if v not in ("none", "unit_power"):
            raise ValueError(
                f"equalize must be 'none' or 'unit_power', got {v!r}"
            )
        return v

    def receive(
        self, field: ComplexArray, dt_s: float | None = None
    ) -> SingleCarrierReceivedSymbols:
        """Demodulate a received field.

        For ``polarization="single"`` the input is 1-D ``(n_samples,)``; for
        ``"dual"`` it is ``(2, n_samples)`` and each polarisation is
        demodulated independently. ``dt_s`` is required only when
        :attr:`cd_compensation_s2` is non-zero (for the frequency grid).
        """
        if self.polarization == "single":
            if field.ndim != 1:
                raise ValueError(
                    f"single-pol receiver expects 1-D field, got shape {field.shape}"
                )
            return self._receive_one_stream(field, dt_s)

        if field.ndim != 2 or field.shape[0] != 2:
            raise ValueError(
                f"dual-pol receiver expects 2-D field with leading axis 2, "
                f"got shape {field.shape}"
            )
        per_pol = [self._receive_one_stream(field[i], dt_s) for i in (0, 1)]
        pilot_syms: ComplexArray | None
        if per_pol[0].pilot_symbols is not None:
            pilot_syms = np.stack(
                [p.pilot_symbols for p in per_pol], axis=0  # type: ignore[misc]
            )
        else:
            pilot_syms = None
        return SingleCarrierReceivedSymbols(
            symbols=np.stack([p.symbols for p in per_pol], axis=0),
            bits=np.stack([p.bits for p in per_pol], axis=0),
            decisions=np.stack([p.decisions for p in per_pol], axis=0),
            modulation_order=self.effective_modulation_order,
            polarization="dual",
            pilots=self.pilots,
            pilot_symbols=pilot_syms,
        )

    def _receive_one_stream(
        self,
        field: ComplexArray,
        dt_s: float | None,
    ) -> SingleCarrierReceivedSymbols:
        """Single 1-D demodulation pipeline."""
        if self.cd_compensation_s2 != 0.0:
            if dt_s is None:
                raise ValueError(
                    "dt_s is required when cd_compensation_s2 != 0 "
                    "(needed for the frequency grid)"
                )
            field = self._undo_dispersion(field, dt_s)

        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        n_samples = field.shape[-1]
        full = np.convolve(field, rrc, mode="full")
        delay = (n_taps - 1) // 2
        filtered = full[delay : delay + n_samples]

        symbols_rx = filtered[:: self.samples_per_symbol]

        if self.equalize == "unit_power":
            avg_power = float(np.mean(np.abs(symbols_rx) ** 2))
            if avg_power > 0:
                symbols_rx = symbols_rx / np.sqrt(avg_power)

        cfg = self._resolved_constellation()
        decisions = nearest_symbol_indices(symbols_rx, cfg.points)
        if self.dm is not None:
            if decisions.size % self.dm.block_length != 0:
                raise ValueError(
                    f"received {decisions.size} symbols, which is not a multiple of "
                    f"dm.block_length ({self.dm.block_length}); pass a field whose "
                    "sample count is n_symbols * samples_per_symbol"
                )
            bits = self.dm.decode(decisions.astype(np.intp))
        else:
            bits = cfg.demap(symbols_rx)

        pilot_symbols: ComplexArray | None = None
        if self.pilots is not None:
            n_sym = symbols_rx.shape[-1]
            pilot_symbols = symbols_rx[self.pilots.pilot_indices(n_sym)]

        return SingleCarrierReceivedSymbols(
            symbols=symbols_rx,
            bits=bits,
            decisions=decisions,
            modulation_order=self.effective_modulation_order,
            polarization="single",
            pilots=self.pilots,
            pilot_symbols=pilot_symbols,
        )

    def _undo_dispersion(self, field: ComplexArray, dt_s: float) -> ComplexArray:
        """Undo accumulated linear chromatic dispersion (β₂ · L) on the field.

        Forward dispersion in :class:`hpcom.backends.numpy.linear.LinearNumpyPropagator`
        applies ``exp(+i β₂/2 ω² L)`` in the frequency domain (NumPy FFT
        convention); the inverse is ``exp(-i β₂/2 ω² L)``.
        """
        n = field.shape[-1]
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt_s)
        spectrum = np.fft.fft(field, axis=-1)
        spectrum = spectrum * np.exp(-1j * self.cd_compensation_s2 / 2.0 * omega**2)
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out
