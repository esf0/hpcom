"""Optical transmitter: random bits → pulse-shaped complex baseband field.

For wavelength-division multiplexing, pass a :class:`hpcom.wdm.WDMConfig` to
``Transmitter.wdm``; the generator then synthesizes one independent baseband
stream per channel, frequency-shifts each onto the WDM grid, and sums them
into a composite wideband field. Per-channel bits and symbols are returned as
2-D arrays of shape ``(n_channels, …)`` so a downstream :class:`Receiver`
configured with the same ``WDMConfig`` and a ``wdm_channel`` index can score
each channel independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import bits_per_symbol, bits_to_symbols, symbols_to_bits
from hpcom.wdm import WDMConfig


@dataclass(frozen=True)
class TransmittedSignal:
    """Output of :meth:`Transmitter.generate`.

    Carries the field plus the original bits/symbols and the parameters needed
    by :class:`hpcom.receiver.Receiver` to demodulate it. When ``wdm`` is set,
    ``bits`` has shape ``(n_channels, n_bits_per_channel)`` and ``symbols``
    has shape ``(n_channels, n_symbols)``; otherwise both are 1-D.
    """

    field: ComplexArray
    dt_s: float
    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    samples_per_symbol: int
    modulation_order: int
    roll_off: float
    pulse_span_symbols: int
    wdm: WDMConfig | None = None


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
    wdm: WDMConfig | None = None

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

        if self.wdm is None:
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
                wdm=None,
            )

        # WDM path: spawn one independent RNG per channel from the same root
        # seed source so reproducibility is identical to single-channel.
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
            wdm=self.wdm,
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
