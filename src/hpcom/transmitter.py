"""Optical transmitter: random bits → pulse-shaped complex baseband field.

The base :class:`Transmitter` is a single-carrier QAM transmitter, optionally
polarisation-multiplexed (``polarization="single"|"dual"``). Subclasses
specialise the signal-shape axis:

* :class:`WDMTransmitter` — wavelength-division multiplexing: N independent
  carriers frequency-shifted onto a uniform grid and summed.
* (future) ``OFDMTransmitter`` — orthogonal-frequency-division multiplexing.

Pilots (:class:`hpcom.pilots.PilotConfig`) are configured on the base class
and apply identically to every stream (per polarisation, per channel for WDM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import bits_per_symbol, bits_to_symbols, symbols_to_bits
from hpcom.pilots import PilotConfig
from hpcom.wdm import WDMConfig

Polarization = Literal["single", "dual"]


@dataclass(frozen=True)
class TransmittedSignal:
    """Output of :meth:`Transmitter.generate`.

    Shape conventions for ``field`` / ``bits`` / ``symbols``:

    * Single-pol, single-carrier: ``field=(n_samples,)``,
      ``bits=(n_symbols*k,)``, ``symbols=(n_symbols,)``.
    * Single-pol, WDM:            ``field=(n_samples,)`` composite,
      ``bits=(n_channels, n_symbols*k)``,
      ``symbols=(n_channels, n_symbols)``.
    * Dual-pol, single-carrier:   ``field=(2, n_samples)``,
      ``bits=(2, n_symbols*k)``, ``symbols=(2, n_symbols)``.
    * Dual-pol, WDM:              ``field=(2, n_samples)`` composite,
      ``bits=(2, n_channels, n_symbols*k)``,
      ``symbols=(2, n_channels, n_symbols)``.

    When ``pilots`` is set, ``symbols`` at pilot positions equal the configured
    pilot value, and the corresponding ``bits`` are zero-filled. Use
    :meth:`data_bits` / :meth:`data_symbols` for the pilot-stripped views.
    """

    field: ComplexArray
    dt_s: float
    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    samples_per_symbol: int
    modulation_order: int
    roll_off: float
    pulse_span_symbols: int
    n_symbols: int
    polarization: Polarization = "single"
    wdm: WDMConfig | None = None
    pilots: PilotConfig | None = None

    @property
    def pilot_indices(self) -> npt.NDArray[np.intp]:
        """Symbol-grid positions of pilots, or empty if no pilots configured."""
        if self.pilots is None:
            return np.empty(0, dtype=np.intp)
        return self.pilots.pilot_indices(self.n_symbols)

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        """Symbol-grid positions of data symbols; all positions if no pilots."""
        if self.pilots is None:
            return np.arange(self.n_symbols, dtype=np.intp)
        return self.pilots.data_indices(self.n_symbols)

    def data_symbols(self) -> ComplexArray:
        """Symbols at data positions only. Preserves leading dims (pol, WDM)."""
        if self.pilots is None:
            return self.symbols
        return self.symbols[..., self.data_indices]

    def data_bits(self) -> npt.NDArray[np.uint8]:
        """Bits at data positions only — Option α convention.

        Preserves the leading dims of :attr:`bits` (pol, WDM channel) and
        shrinks the trailing bit axis by the pilot fraction.
        """
        if self.pilots is None:
            return self.bits
        k = bits_per_symbol(self.modulation_order)
        idx = self.data_indices
        reshaped = self.bits.reshape(*self.bits.shape[:-1], -1, k)
        selected = reshaped[..., idx, :]
        return selected.reshape(*self.bits.shape[:-1], -1)


@dataclass(frozen=True)
class Transmitter:
    """Single-carrier QAM optical transmitter (optionally dual-polarisation).

    For WDM (multiple carriers), use :class:`WDMTransmitter`. For pilot
    insertion, pass a :class:`~hpcom.pilots.PilotConfig`. For PDM (X/Y), set
    ``polarization="dual"`` — bits/symbols/field gain a leading axis of size 2.
    """

    modulation_order: int
    symbol_rate_hz: float
    n_symbols: int
    samples_per_symbol: int = 16
    roll_off: float = 0.1
    pulse_span_symbols: int = 64
    seed: int | None = None
    polarization: Polarization = "single"
    pilots: PilotConfig | None = None

    def __post_init__(self) -> None:
        if self.n_symbols < 1:
            raise ValueError(f"n_symbols must be >= 1, got {self.n_symbols}")
        if self.symbol_rate_hz <= 0:
            raise ValueError(f"symbol_rate_hz must be > 0, got {self.symbol_rate_hz}")
        if self.samples_per_symbol < 2:
            raise ValueError(
                f"samples_per_symbol must be >= 2 (Nyquist), got {self.samples_per_symbol}"
            )
        bits_per_symbol(self.modulation_order)

        if self.polarization not in ("single", "dual"):
            raise ValueError(
                f"polarization must be 'single' or 'dual', got {self.polarization!r}"
            )

    @property
    def dt_s(self) -> float:
        """Sample period of the generated field in seconds."""
        return 1.0 / (self.symbol_rate_hz * self.samples_per_symbol)

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> TransmittedSignal:
        """Generate a transmitted signal.

        ``rng`` overrides the configured ``seed`` (useful when a parent
        Simulation threads reproducibility across stages).

        For deterministic inputs (testing, replay, custom shaping) pass either
        ``bits`` or ``symbols``. Shapes for single-carrier:

        - single-pol: ``bits=(n_symbols*k,)``, ``symbols=(n_symbols,)``.
        - dual-pol:  ``bits=(2, n_symbols*k)``, ``symbols=(2, n_symbols)``.

        ``bits`` and ``symbols`` are mutually exclusive.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        if self.polarization == "single":
            single_rng = rng if rng is not None else np.random.default_rng(self.seed)
            ch_bits, ch_syms, field = self._generate_one_channel(
                single_rng,
                bits=_as_1d(bits, "bits") if bits is not None else None,
                symbols=_as_1d(symbols, "symbols") if symbols is not None else None,
            )
            return self._make_signal(
                field=field, symbols=ch_syms, bits=ch_bits,
                polarization="single", wdm=None,
            )

        # Dual-pol single-carrier.
        k = bits_per_symbol(self.modulation_order)
        if bits is not None:
            b2 = np.asarray(bits, dtype=np.uint8)
            if b2.shape != (2, self.n_symbols * k):
                raise ValueError(
                    f"dual-pol `bits` must have shape (2, {self.n_symbols * k}), "
                    f"got {b2.shape}"
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
            p_bits, p_syms, p_field = self._generate_one_channel(
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
            wdm=None,
        )

    def _make_signal(
        self,
        *,
        field: ComplexArray,
        symbols: ComplexArray,
        bits: npt.NDArray[np.uint8],
        polarization: Polarization,
        wdm: WDMConfig | None,
    ) -> TransmittedSignal:
        """Bundle a ``TransmittedSignal`` with this Tx's metadata."""
        return TransmittedSignal(
            field=field,
            dt_s=self.dt_s,
            symbols=symbols,
            bits=bits,
            samples_per_symbol=self.samples_per_symbol,
            modulation_order=self.modulation_order,
            roll_off=self.roll_off,
            pulse_span_symbols=self.pulse_span_symbols,
            n_symbols=self.n_symbols,
            polarization=polarization,
            wdm=wdm,
            pilots=self.pilots,
        )

    def _generate_one_channel(
        self,
        rng: np.random.Generator,
        *,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray, ComplexArray]:
        """Generate ``(bits, symbols, baseband-field)`` for one 1-D stream.

        Subclasses orchestrate multiple calls to this method to build their
        composite output (e.g. one call per WDM channel × polarisation).
        """
        k = bits_per_symbol(self.modulation_order)
        if symbols is not None:
            ch_symbols = np.asarray(symbols, dtype=np.complex128)
            if ch_symbols.shape != (self.n_symbols,):
                raise ValueError(
                    f"`symbols` must have shape ({self.n_symbols},), "
                    f"got {ch_symbols.shape}"
                )
            ch_bits = symbols_to_bits(ch_symbols, self.modulation_order)
        elif bits is not None:
            ch_bits = np.asarray(bits, dtype=np.uint8)
            if ch_bits.shape != (self.n_symbols * k,):
                raise ValueError(
                    f"`bits` must have shape ({self.n_symbols * k},), "
                    f"got {ch_bits.shape}"
                )
            ch_symbols = np.asarray(
                bits_to_symbols(ch_bits, self.modulation_order), dtype=np.complex128
            )
        else:
            ch_bits = rng.integers(0, 2, size=self.n_symbols * k, dtype=np.uint8)
            ch_symbols = np.asarray(
                bits_to_symbols(ch_bits, self.modulation_order), dtype=np.complex128
            )

        if self.pilots is not None:
            pilot_idx = self.pilots.pilot_indices(self.n_symbols)
            ch_symbols = ch_symbols.copy()
            ch_symbols[pilot_idx] = self.pilots.symbol
            ch_bits = ch_bits.copy()
            for p in pilot_idx:
                ch_bits[int(p) * k : (int(p) + 1) * k] = 0

        n_samples = self.n_symbols * self.samples_per_symbol
        upsampled = np.zeros(n_samples, dtype=np.complex128)
        upsampled[:: self.samples_per_symbol] = ch_symbols

        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        full = np.convolve(upsampled, rrc, mode="full")
        delay = (n_taps - 1) // 2
        field = full[delay : delay + n_samples].astype(np.complex128)
        return ch_bits, ch_symbols, field


@dataclass(frozen=True)
class WDMTransmitter(Transmitter):
    """Wavelength-division-multiplexed transmitter.

    Generates ``wdm.n_channels`` independent QAM streams (per polarisation),
    frequency-shifts each onto the configured grid, and sums them into a
    composite wideband field. With ``polarization="dual"``, each polarisation
    independently carries the full N-channel composite; X and Y streams are
    independent (PDM-WDM).
    """

    wdm: WDMConfig = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Wideband Nyquist: sample rate must cover the outer channel edges.
        # Per-channel one-sided bandwidth is Rs · (1 + roll_off) / 2.
        sample_rate_hz = self.symbol_rate_hz * self.samples_per_symbol
        outer_center = abs(self.wdm.channel_frequencies_hz()[-1])
        needed = 2.0 * (
            outer_center + self.symbol_rate_hz * (1.0 + self.roll_off) / 2.0
        )
        if sample_rate_hz < needed:
            raise ValueError(
                f"WDM wideband sample rate ({sample_rate_hz:.3e} Hz) is below "
                f"the Nyquist requirement ({needed:.3e} Hz) for "
                f"{self.wdm.n_channels} channels at {self.wdm.spacing_hz:.3e} Hz "
                f"spacing with roll-off {self.roll_off}. Increase "
                f"samples_per_symbol (or reduce spacing/channel count)."
            )

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> TransmittedSignal:
        """Generate the WDM composite field.

        Deterministic-input shapes (single carrier per polarisation per channel):

        - single-pol: ``bits=(n_channels, n_symbols*k)``,
          ``symbols=(n_channels, n_symbols)``.
        - dual-pol:   ``bits=(2, n_channels, n_symbols*k)``,
          ``symbols=(2, n_channels, n_symbols)``.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        n_ch = self.wdm.n_channels
        k = bits_per_symbol(self.modulation_order)
        n_pol = 2 if self.polarization == "dual" else 1

        # Validate deterministic-input shapes.
        if bits is not None:
            b = np.asarray(bits, dtype=np.uint8)
            expected = (
                (2, n_ch, self.n_symbols * k) if n_pol == 2 else (n_ch, self.n_symbols * k)
            )
            if b.shape != expected:
                raise ValueError(
                    f"`bits` must have shape {expected}, got {b.shape}"
                )
        if symbols is not None:
            s = np.asarray(symbols, dtype=np.complex128)
            expected_s = (
                (2, n_ch, self.n_symbols) if n_pol == 2 else (n_ch, self.n_symbols)
            )
            if s.shape != expected_s:
                raise ValueError(
                    f"`symbols` must have shape {expected_s}, got {s.shape}"
                )

        # Seed tree: root → per-pol → per-channel. Identical to single-carrier
        # dual-pol when n_ch=1.
        seed_seq = _seed_seq_from(rng, self.seed)
        pol_seqs = seed_seq.spawn(n_pol) if n_pol == 2 else [seed_seq]

        n_samples = self.n_symbols * self.samples_per_symbol
        t = np.arange(n_samples) * self.dt_s
        freqs = self.wdm.channel_frequencies_hz()

        per_pol_field: list[ComplexArray] = []
        per_pol_bits: list[npt.NDArray[np.uint8]] = []
        per_pol_syms: list[ComplexArray] = []

        for pol_idx, pol_seq in enumerate(pol_seqs):
            ch_sub_seqs = pol_seq.spawn(n_ch)
            composite = np.zeros(n_samples, dtype=np.complex128)
            this_pol_bits: list[npt.NDArray[np.uint8]] = []
            this_pol_syms: list[ComplexArray] = []
            for ch_idx, (f_ch, sub_seq) in enumerate(
                zip(freqs, ch_sub_seqs, strict=True)
            ):
                ch_rng = np.random.default_rng(sub_seq)
                if bits is not None:
                    b_in = bits[pol_idx, ch_idx] if n_pol == 2 else bits[ch_idx]
                else:
                    b_in = None
                if symbols is not None:
                    s_in = symbols[pol_idx, ch_idx] if n_pol == 2 else symbols[ch_idx]
                else:
                    s_in = None
                ch_bits, ch_syms, field_ch = self._generate_one_channel(
                    ch_rng, bits=b_in, symbols=s_in
                )
                this_pol_bits.append(ch_bits)
                this_pol_syms.append(ch_syms)
                composite += field_ch * np.exp(2j * np.pi * float(f_ch) * t)
            per_pol_field.append(composite)
            per_pol_bits.append(np.stack(this_pol_bits, axis=0))
            per_pol_syms.append(np.stack(this_pol_syms, axis=0))

        if n_pol == 1:
            return self._make_signal(
                field=per_pol_field[0],
                symbols=per_pol_syms[0],
                bits=per_pol_bits[0],
                polarization="single",
                wdm=self.wdm,
            )

        return self._make_signal(
            field=np.stack(per_pol_field, axis=0),
            symbols=np.stack(per_pol_syms, axis=0),
            bits=np.stack(per_pol_bits, axis=0),
            polarization="dual",
            wdm=self.wdm,
        )


def _as_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Validate that a deterministic-input array is 1-D (single-channel path)."""
    if arr.ndim != 1:
        raise ValueError(
            f"`{name}` must be 1-D for a single-channel Transmitter, got {arr.ndim}-D"
        )
    return arr


def _seed_seq_from(
    rng: np.random.Generator | None, seed: int | None
) -> np.random.SeedSequence:
    """Build a reproducible SeedSequence from either a generator or a seed."""
    if rng is not None:
        return np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
    return np.random.SeedSequence(seed)
