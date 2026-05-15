"""Optical transmitter: random bits → pulse-shaped complex baseband field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import bits_per_symbol, bits_to_symbols


@dataclass(frozen=True)
class TransmittedSignal:
    """Output of :meth:`Transmitter.generate`.

    Carries the field plus the original bits/symbols and the parameters needed
    by :class:`hpcom.receiver.Receiver` to demodulate it.
    """

    field: ComplexArray
    dt_s: float
    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    samples_per_symbol: int
    modulation_order: int
    roll_off: float
    pulse_span_symbols: int


@dataclass(frozen=True)
class Transmitter:
    """Single-polarization, single-channel optical transmitter.

    Generates a random bit stream, maps it to QAM symbols, upsamples to the
    sample grid and pulse-shapes with a root-raised-cosine filter. Output
    field is complex baseband at sample period ``1 / (symbol_rate_hz · sps)``.
    """

    modulation_order: int
    symbol_rate_hz: float
    n_symbols: int
    samples_per_symbol: int = 16
    roll_off: float = 0.1
    pulse_span_symbols: int = 64
    seed: int | None = None

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
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        k = bits_per_symbol(self.modulation_order)
        bits = rng.integers(0, 2, size=self.n_symbols * k, dtype=np.uint8)
        symbols = bits_to_symbols(bits, self.modulation_order)

        n_samples = self.n_symbols * self.samples_per_symbol
        upsampled = np.zeros(n_samples, dtype=np.complex128)
        upsampled[:: self.samples_per_symbol] = symbols

        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        # Full convolution then trim so symbol m sits at index m * sps in the
        # output (RRC peak delay (n_taps-1)//2 is removed).
        full = np.convolve(upsampled, rrc, mode="full")
        delay = (n_taps - 1) // 2
        field = full[delay : delay + n_samples].astype(np.complex128)

        return TransmittedSignal(
            field=field,
            dt_s=self.dt_s,
            symbols=symbols,
            bits=bits,
            samples_per_symbol=self.samples_per_symbol,
            modulation_order=self.modulation_order,
            roll_off=self.roll_off,
            pulse_span_symbols=self.pulse_span_symbols,
        )
