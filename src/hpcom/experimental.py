"""Skeleton for prototyping custom waveforms.

This module gives you a minimal, runnable Transmitter/Receiver pair that you
can copy and modify to define a new signal shape — anything that doesn't
already fit :class:`~hpcom.transmitter.SingleCarrierTransmitter`,
:class:`~hpcom.wdm.WDMTransmitter`, or :class:`~hpcom.ofdm.OFDMTransmitter`.

What this module deliberately does *not* do:

* It does not modify or extend :class:`~hpcom.simulation.Simulation`. That
  class is contracted to single-carrier / WDM today. To drive a custom
  waveform end-to-end, call :meth:`~hpcom.link.Link.propagate` directly (see
  ``examples/05_custom_waveform.py``).
* It does not try to unify "every possible waveform" into one base class.
  Different waveforms have genuinely different parameters; copy this file
  and adapt rather than fight the type system.

Contract a new waveform must satisfy
------------------------------------

A custom Transmitter must subclass :class:`~hpcom.transmitter.Transmitter`
and provide:

1. The ``dt_s`` property — sample period of the produced field, in seconds.
2. A ``generate(...)`` method returning a :class:`TransmittedSignal` subclass
   carrying ``field``, ``dt_s``, ``symbols``, ``bits``, ``n_symbols``,
   ``polarization`` (and ``pilots`` if relevant).
3. Field shape: ``(n_samples,)`` for ``polarization="single"``;
   ``(2, n_samples)`` for ``"dual"``. The same axis convention applies to
   ``bits`` and ``symbols``.

A matching Receiver must subclass :class:`~hpcom.receiver.Receiver` and
implement ``receive(field, dt_s=None) -> ReceivedSymbols``.

The default implementation below is a deliberately *trivial* waveform:
rectangular-pulse QAM (sample-and-hold, no filtering). It is **not** a good
waveform — it has wide spectral side-lobes and zero CD robustness. Its only
purpose is to demonstrate the integration contract so you can swap in the
real shaping/equalisation code with confidence that everything else (link
propagation, polarisation handling, BER scoring) works.

Where to put your real DSP code: each block of the default ``generate`` /
``receive`` body marked ``# >>> REPLACE THIS:`` is where waveform-specific
logic lives.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, Field

from hpcom._typing import ComplexArray
from hpcom.modulation import nearest_symbol_indices
from hpcom.receiver import ReceivedSymbols, Receiver
from hpcom.transmitter import (
    Polarization,
    TransmittedSignal,
    Transmitter,
    _as_1d,
    _seed_seq_from,
)


class ExperimentalSignal(TransmittedSignal):
    """Output of :class:`ExperimentalTransmission`.

    Add fields here for whatever waveform-specific metadata downstream code
    (your custom receiver, plotting helpers, …) needs to see. The base
    already carries ``field``, ``dt_s``, ``symbols``, ``bits``, ``n_symbols``,
    ``polarization``, ``pilots``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    samples_per_symbol: int
    """Sample-rate vs. symbol-rate ratio used by the trivial default. Drop or
    rename this field if your waveform has no such concept."""


class ExperimentalTransmission(Transmitter):
    """Trivial reference waveform: rectangular-pulse QAM, sample-and-hold.

    Field is built by repeating each QAM symbol ``samples_per_symbol`` times
    in the time domain — no RRC, no spectral shaping. Survives noiseless
    back-to-back but degrades fast on a dispersive fibre because of its
    sinc-shaped spectrum.

    Copy this class, give it a distinctive name, and adapt the ``generate``
    body for your waveform.
    """

    samples_per_symbol: int = Field(default=8, ge=1)
    """Number of time-samples emitted per QAM symbol. Rename / replace with
    your waveform's natural oversampling parameter (e.g. an IFFT size, a DAC
    interpolation factor)."""

    @property
    def dt_s(self) -> float:
        # >>> REPLACE THIS: return the sample period of your field.
        return 1.0 / (self.symbol_rate_hz * self.samples_per_symbol)

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> ExperimentalSignal:
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        if self.polarization == "single":
            single_rng = rng if rng is not None else np.random.default_rng(self.seed)
            ch_bits, ch_syms, field = self._one_stream(
                single_rng,
                bits=_as_1d(bits, "bits") if bits is not None else None,
                symbols=_as_1d(symbols, "symbols") if symbols is not None else None,
            )
            return self._make_signal(
                field=field, symbols=ch_syms, bits=ch_bits, polarization="single",
            )

        # Dual-pol: two independently-seeded streams stacked on axis 0.
        bps = self._bits_per_stream()
        if bits is not None:
            b2 = np.asarray(bits, dtype=np.uint8)
            if b2.shape != (2, bps):
                raise ValueError(
                    f"dual-pol `bits` must have shape (2, {bps}), got {b2.shape}"
                )
        if symbols is not None:
            s2 = np.asarray(symbols, dtype=np.complex128)
            if s2.shape != (2, self.n_symbols):
                raise ValueError(
                    f"dual-pol `symbols` must have shape (2, {self.n_symbols}), "
                    f"got {s2.shape}"
                )

        seed_seq = _seed_seq_from(rng, self.seed)
        pol_seqs = seed_seq.spawn(2)
        per_pol_bits: list[npt.NDArray[np.uint8]] = []
        per_pol_syms: list[ComplexArray] = []
        per_pol_fields: list[ComplexArray] = []
        for pol_idx, sub_seq in enumerate(pol_seqs):
            pol_rng = np.random.default_rng(sub_seq)
            p_bits, p_syms, p_field = self._one_stream(
                pol_rng,
                bits=bits[pol_idx] if bits is not None else None,
                symbols=symbols[pol_idx] if symbols is not None else None,
            )
            per_pol_bits.append(p_bits)
            per_pol_syms.append(p_syms)
            per_pol_fields.append(p_field)

        return self._make_signal(
            field=np.stack(per_pol_fields, axis=0),
            symbols=np.stack(per_pol_syms, axis=0),
            bits=np.stack(per_pol_bits, axis=0),
            polarization="dual",
        )

    def _one_stream(
        self,
        rng: np.random.Generator,
        *,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray, ComplexArray]:
        """Produce ``(bits, symbols, field)`` for one polarisation stream."""
        cfg = self._resolved_constellation()
        n_bits = self._bits_per_stream()
        dm = self.dm

        # ---- bits/symbols (shared bit-mapping plumbing) --------------------
        if symbols is not None:
            ch_symbols = np.asarray(symbols, dtype=np.complex128)
            if dm is not None:
                indices = nearest_symbol_indices(ch_symbols, cfg.points).astype(np.intp)
                ch_bits = dm.decode(indices)
            else:
                ch_bits = cfg.demap(ch_symbols)
        elif bits is not None:
            ch_bits = np.asarray(bits, dtype=np.uint8)
            if dm is not None:
                indices = dm.encode(ch_bits)
                ch_symbols = cfg.points[indices].astype(np.complex128)
            else:
                ch_symbols = np.asarray(cfg.map(ch_bits), dtype=np.complex128)
        else:
            ch_bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
            if dm is not None:
                indices = dm.encode(ch_bits)
                ch_symbols = cfg.points[indices].astype(np.complex128)
            else:
                ch_symbols = np.asarray(cfg.map(ch_bits), dtype=np.complex128)

        # >>> REPLACE THIS: build the baseband ``field`` from ``ch_symbols``.
        # The default is rectangular pulse shaping (zero-order hold): each
        # QAM symbol held for ``samples_per_symbol`` consecutive samples.
        field = np.repeat(ch_symbols, self.samples_per_symbol).astype(np.complex128)

        return ch_bits, ch_symbols, field

    def _make_signal(
        self,
        *,
        field: ComplexArray,
        symbols: ComplexArray,
        bits: npt.NDArray[np.uint8],
        polarization: Polarization,
    ) -> ExperimentalSignal:
        return ExperimentalSignal(
            field=field,
            dt_s=self.dt_s,
            symbols=symbols,
            bits=bits,
            n_symbols=self.n_symbols,
            samples_per_symbol=self.samples_per_symbol,
            polarization=polarization,
            pilots=self.pilots,
        )


class ExperimentalReceivedSymbols(ReceivedSymbols):
    """Matching :class:`ReceivedSymbols` subclass — add fields as needed."""


class ExperimentalReceiver(Receiver):
    """Matching receiver for :class:`ExperimentalTransmission`.

    Decimates by ``samples_per_symbol`` (no matched filter) and hard-decides
    to the nearest constellation point. Replace ``receive`` with your real
    DSP pipeline (matched filter, CD compensation, equaliser, carrier-phase
    recovery, …) when you graduate from the trivial waveform.
    """

    samples_per_symbol: int = Field(default=8, ge=1)
    """Must match the transmitter's value."""

    def receive(
        self, field: ComplexArray, dt_s: float | None = None
    ) -> ExperimentalReceivedSymbols:
        del dt_s  # unused; the trivial waveform has no per-sample timing logic.

        if self.polarization == "single":
            if field.ndim != 1:
                raise ValueError(
                    f"single-pol receiver expects 1-D field, got shape {field.shape}"
                )
            return self._decode_one(field, polarization="single")

        if field.ndim != 2 or field.shape[0] != 2:
            raise ValueError(
                f"dual-pol receiver expects 2-D field with leading axis 2, "
                f"got shape {field.shape}"
            )
        x = self._decode_one(field[0], polarization="dual", _stack=True)
        y = self._decode_one(field[1], polarization="dual", _stack=True)
        return ExperimentalReceivedSymbols(
            symbols=np.stack([x.symbols, y.symbols], axis=0),
            bits=np.stack([x.bits, y.bits], axis=0),
            decisions=np.stack([x.decisions, y.decisions], axis=0),
            modulation_order=self.effective_modulation_order,
            polarization="dual",
            pilots=self.pilots,
        )

    def _decode_one(
        self,
        field: ComplexArray,
        *,
        polarization: Polarization,
        _stack: bool = False,
    ) -> ExperimentalReceivedSymbols:
        cfg = self._resolved_constellation()

        # >>> REPLACE THIS: recover symbol-rate samples from ``field``.
        # The default takes one sample per symbol period (the centre of each
        # sample-and-hold block). Real receivers do matched filtering and
        # CD-compensation here first.
        offset = self.samples_per_symbol // 2
        rx_syms = field[offset :: self.samples_per_symbol].astype(np.complex128)

        # Hard decision to nearest constellation point.
        decisions = nearest_symbol_indices(rx_syms, cfg.points).astype(np.intp)
        decided_syms = cfg.points[decisions].astype(np.complex128)

        bits = (
            self.dm.decode(decisions) if self.dm is not None else cfg.demap(decided_syms)
        )

        return ExperimentalReceivedSymbols(
            symbols=rx_syms,
            bits=bits,
            decisions=decisions,
            modulation_order=self.effective_modulation_order,
            polarization=polarization,
            pilots=self.pilots,
        )


__all__ = [
    "ExperimentalReceivedSymbols",
    "ExperimentalReceiver",
    "ExperimentalSignal",
    "ExperimentalTransmission",
]
