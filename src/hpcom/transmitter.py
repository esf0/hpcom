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
from hpcom.modulation import bits_per_symbol, bits_to_symbols
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
        self, *, rng: np.random.Generator | None = None
    ) -> TransmittedSignal:
        """Generate a transmitted signal.

        ``rng`` overrides the configured ``seed``: pass an explicit generator
        when a parent simulation needs to thread reproducibility across stages.
        For WDM, each channel draws from an independent sub-RNG spawned from
        the same seed/rng so single- vs multi-channel runs share the seed
        contract.
        """
        if self.wdm is None:
            single_rng = rng if rng is not None else np.random.default_rng(self.seed)
            bits, symbols, field = self._generate_one_channel(single_rng)
            return TransmittedSignal(
                field=field,
                dt_s=self.dt_s,
                symbols=symbols,
                bits=bits,
                samples_per_symbol=self.samples_per_symbol,
                modulation_order=self.modulation_order,
                roll_off=self.roll_off,
                pulse_span_symbols=self.pulse_span_symbols,
                wdm=None,
            )

        # WDM path: spawn one independent RNG per channel from the same root
        # seed source so reproducibility is identical to single-channel.
        if rng is not None:
            # default_rng exposes its SeedSequence via .bit_generator.seed_seq
            # only in NumPy >=1.25; spawn from a fresh sequence keyed by a
            # random uint64 drawn from `rng` so we don't depend on that.
            seed_seq = np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
        else:
            seed_seq = np.random.SeedSequence(self.seed)
        sub_seqs = seed_seq.spawn(self.wdm.n_channels)

        n_samples = self.n_symbols * self.samples_per_symbol
        t = np.arange(n_samples) * self.dt_s
        composite = np.zeros(n_samples, dtype=np.complex128)
        per_ch_bits: list[npt.NDArray[np.uint8]] = []
        per_ch_symbols: list[ComplexArray] = []
        freqs = self.wdm.channel_frequencies_hz()

        for f_ch, sub_seq in zip(freqs, sub_seqs, strict=True):
            ch_rng = np.random.default_rng(sub_seq)
            bits_ch, syms_ch, field_ch = self._generate_one_channel(ch_rng)
            per_ch_bits.append(bits_ch)
            per_ch_symbols.append(syms_ch)
            # Frequency-shift channel from baseband to its WDM slot.
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
        self, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray, ComplexArray]:
        """Generate (bits, symbols, baseband-field) for one channel."""
        k = bits_per_symbol(self.modulation_order)
        bits = rng.integers(0, 2, size=self.n_symbols * k, dtype=np.uint8)
        symbols = bits_to_symbols(bits, self.modulation_order)

        n_samples = self.n_symbols * self.samples_per_symbol
        upsampled = np.zeros(n_samples, dtype=np.complex128)
        upsampled[:: self.samples_per_symbol] = symbols

        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        full = np.convolve(upsampled, rrc, mode="full")
        delay = (n_taps - 1) // 2
        field = full[delay : delay + n_samples].astype(np.complex128)
        return bits, symbols, field
