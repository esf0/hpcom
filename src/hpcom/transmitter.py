"""Optical transmitter: random bits → pulse-shaped complex baseband field.

Three independent feature axes compose:

* **Polarisation** — ``polarization="single"`` returns a 1-D field; ``"dual"``
  returns a ``(2, n_samples)`` field with independent X/Y bit streams (PDM).
* **WDM** — when :attr:`wdm` is set, each channel is independently generated
  and frequency-shifted onto the grid before summation. Bits/symbols gain a
  leading ``n_channels`` axis. (Dual-pol × WDM is handled by a subclass —
  see :class:`hpcom.transmitter.WDMTransmitter` once it lands.)
* **Pilots** — when :attr:`pilots` is set, every ``period``-th symbol carries
  the configured reference value; the same pilot pattern is applied to every
  stream (per polarisation, per WDM channel).
"""

from __future__ import annotations

from dataclasses import dataclass
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

    * Single-pol, no WDM: ``field=(n_samples,)``, ``bits=(n_symbols*k,)``,
      ``symbols=(n_symbols,)``.
    * Single-pol, WDM: ``field=(n_samples,)`` composite,
      ``bits=(n_channels, n_symbols*k)``, ``symbols=(n_channels, n_symbols)``.
    * Dual-pol, no WDM: ``field=(2, n_samples)``,
      ``bits=(2, n_symbols*k)``, ``symbols=(2, n_symbols)``.

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
        """Symbols at data positions only (1-D for single-channel, 2-D for WDM)."""
        if self.pilots is None:
            return self.symbols
        idx = self.data_indices
        return self.symbols[..., idx]

    def data_bits(self) -> npt.NDArray[np.uint8]:
        """Bits at data positions only — Option α convention.

        Shape follows the underlying ``bits``: 1-D ``(n_data * k,)`` for
        single-channel or 2-D ``(n_channels, n_data * k)`` for WDM.
        """
        if self.pilots is None:
            return self.bits
        k = bits_per_symbol(self.modulation_order)
        idx = self.data_indices
        if self.bits.ndim == 1:
            return self.bits.reshape(-1, k)[idx].ravel()
        # WDM: per-channel slice.
        per_ch = self.bits.reshape(self.bits.shape[0], -1, k)
        return per_ch[:, idx, :].reshape(self.bits.shape[0], -1)


@dataclass(frozen=True)
class Transmitter:
    """Single-polarization optical transmitter, optionally WDM-multiplexed.

    Generates a random bit stream per channel, maps it to QAM symbols,
    upsamples to the sample grid and pulse-shapes with a root-raised-cosine
    filter. When :attr:`wdm` is set, each channel is independently generated
    and frequency-shifted onto the WDM grid before summation. Output field is
    complex baseband at sample period ``1 / (symbol_rate_hz · sps)``.
    """

    modulation_order: int
    symbol_rate_hz: float
    n_symbols: int
    samples_per_symbol: int = 16
    roll_off: float = 0.1
    pulse_span_symbols: int = 64
    seed: int | None = None
    polarization: Polarization = "single"
    wdm: WDMConfig | None = None
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
        # Validate modulation order via the canonical helper.
        bits_per_symbol(self.modulation_order)

        if self.polarization not in ("single", "dual"):
            raise ValueError(
                f"polarization must be 'single' or 'dual', got {self.polarization!r}"
            )
        if self.polarization == "dual" and self.wdm is not None:
            raise NotImplementedError(
                "dual-pol × WDM is handled by WDMTransmitter (planned subclass); "
                "for now use polarization='single' when wdm is set."
            )

        if self.wdm is not None:
            # Wideband Nyquist: sample rate must cover the outer channel edges.
            # Per-channel one-sided bandwidth is Rs · (1 + roll_off) / 2.
            sample_rate_hz = self.symbol_rate_hz * self.samples_per_symbol
            outer_center = abs(self.wdm.channel_frequencies_hz()[-1])
            needed = 2.0 * (outer_center + self.symbol_rate_hz * (1.0 + self.roll_off) / 2.0)
            if sample_rate_hz < needed:
                raise ValueError(
                    f"WDM wideband sample rate ({sample_rate_hz:.3e} Hz) is below "
                    f"the Nyquist requirement ({needed:.3e} Hz) for "
                    f"{self.wdm.n_channels} channels at {self.wdm.spacing_hz:.3e} Hz "
                    f"spacing with roll-off {self.roll_off}. Increase "
                    f"samples_per_symbol (or reduce spacing/channel count)."
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

        ``rng`` overrides the configured ``seed``: pass an explicit generator
        when a parent simulation needs to thread reproducibility across stages.
        For WDM, each channel draws from an independent sub-RNG spawned from
        the same seed/rng so single- vs multi-channel runs share the seed
        contract.

        For deterministic inputs (testing, replay, custom shaping) pass either
        ``bits`` (will be mapped to constellation symbols) or ``symbols``
        directly (used as-is, no constellation check; the returned ``bits``
        are the hard-decision demapping). Shapes:

        - single channel: ``bits`` is ``(n_symbols * k,)``,
          ``symbols`` is ``(n_symbols,)``.
        - WDM: leading axis is ``n_channels``.

        ``bits`` and ``symbols`` are mutually exclusive.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        if self.wdm is None and self.polarization == "single":
            single_rng = rng if rng is not None else np.random.default_rng(self.seed)
            ch_bits, ch_syms, field = self._generate_one_channel(
                single_rng,
                bits=_as_1d(bits, "bits") if bits is not None else None,
                symbols=_as_1d(symbols, "symbols") if symbols is not None else None,
            )
            return TransmittedSignal(
                field=field,
                dt_s=self.dt_s,
                symbols=ch_syms,
                bits=ch_bits,
                samples_per_symbol=self.samples_per_symbol,
                modulation_order=self.modulation_order,
                roll_off=self.roll_off,
                pulse_span_symbols=self.pulse_span_symbols,
                n_symbols=self.n_symbols,
                polarization="single",
                wdm=None,
                pilots=self.pilots,
            )

        if self.polarization == "dual":
            # Dual-pol, no WDM. Two independent streams (X and Y), one
            # sub-RNG each, stacked into (2, n_samples). Pilots and
            # deterministic bits/symbols are applied per pol.
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

            if rng is not None:
                seed_seq = np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
            else:
                seed_seq = np.random.SeedSequence(self.seed)
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

            return TransmittedSignal(
                field=np.stack(per_pol_fields, axis=0),
                dt_s=self.dt_s,
                symbols=np.stack(per_pol_syms, axis=0),
                bits=np.stack(per_pol_bits, axis=0),
                samples_per_symbol=self.samples_per_symbol,
                modulation_order=self.modulation_order,
                roll_off=self.roll_off,
                pulse_span_symbols=self.pulse_span_symbols,
                n_symbols=self.n_symbols,
                polarization="dual",
                wdm=None,
                pilots=self.pilots,
            )

        # WDM path: spawn one independent RNG per channel from the same root
        # seed source so reproducibility is identical to single-channel.
        # (single-pol × WDM only; dual-pol × WDM was rejected in __post_init__.)
        assert self.wdm is not None
        n_ch = self.wdm.n_channels
        if bits is not None:
            bits_2d = np.asarray(bits, dtype=np.uint8)
            if bits_2d.shape[0] != n_ch:
                raise ValueError(
                    f"WDM `bits` leading axis must equal n_channels ({n_ch}), "
                    f"got shape {bits_2d.shape}"
                )
        if symbols is not None:
            sym_2d = np.asarray(symbols, dtype=np.complex128)
            if sym_2d.shape[0] != n_ch:
                raise ValueError(
                    f"WDM `symbols` leading axis must equal n_channels ({n_ch}), "
                    f"got shape {sym_2d.shape}"
                )

        if rng is not None:
            seed_seq = np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
        else:
            seed_seq = np.random.SeedSequence(self.seed)
        sub_seqs = seed_seq.spawn(n_ch)

        n_samples = self.n_symbols * self.samples_per_symbol
        t = np.arange(n_samples) * self.dt_s
        composite = np.zeros(n_samples, dtype=np.complex128)
        per_ch_bits: list[npt.NDArray[np.uint8]] = []
        per_ch_symbols: list[ComplexArray] = []
        freqs = self.wdm.channel_frequencies_hz()

        for ch_idx, (f_ch, sub_seq) in enumerate(zip(freqs, sub_seqs, strict=True)):
            ch_rng = np.random.default_rng(sub_seq)
            ch_bits, ch_syms, field_ch = self._generate_one_channel(
                ch_rng,
                bits=bits[ch_idx] if bits is not None else None,
                symbols=symbols[ch_idx] if symbols is not None else None,
            )
            per_ch_bits.append(ch_bits)
            per_ch_symbols.append(ch_syms)
            composite += field_ch * np.exp(2j * np.pi * float(f_ch) * t)

        return TransmittedSignal(
            field=composite,
            dt_s=self.dt_s,
            symbols=np.stack(per_ch_symbols, axis=0),
            bits=np.stack(per_ch_bits, axis=0),
            samples_per_symbol=self.samples_per_symbol,
            modulation_order=self.modulation_order,
            roll_off=self.roll_off,
            pulse_span_symbols=self.pulse_span_symbols,
            n_symbols=self.n_symbols,
            wdm=self.wdm,
            pilots=self.pilots,
        )

    def _generate_one_channel(
        self,
        rng: np.random.Generator,
        *,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray, ComplexArray]:
        """Generate (bits, symbols, baseband-field) for one channel."""
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
            ch_symbols = np.asarray(bits_to_symbols(ch_bits, self.modulation_order), dtype=np.complex128)
        else:
            ch_bits = rng.integers(0, 2, size=self.n_symbols * k, dtype=np.uint8)
            ch_symbols = np.asarray(bits_to_symbols(ch_bits, self.modulation_order), dtype=np.complex128)

        # Insert pilots after data generation: overwrite symbols at pilot
        # positions, and zero-fill the corresponding bits (they are not real
        # data; data_bits() strips them).
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


def _as_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Validate that a deterministic-input array is 1-D (single-channel path)."""
    if arr.ndim != 1:
        raise ValueError(
            f"`{name}` must be 1-D for a single-channel Transmitter, got {arr.ndim}-D"
        )
    return arr
