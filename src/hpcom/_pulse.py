"""Pulse-shaping filters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def root_raised_cosine(
    n_taps: int, samples_per_symbol: int, roll_off: float
) -> npt.NDArray[np.float64]:
    """Root-raised-cosine FIR with unit *energy* (Σ h² = 1).

    Two cascaded unit-energy RRC filters yield a raised-cosine response that is
    Nyquist-sampled at integer multiples of the symbol period — i.e. zero ISI
    when the receiver matched-filters and downsamples at the symbol rate.

    Parameters
    ----------
    n_taps
        Filter length in samples. Should be odd so the peak sits on a sample.
    samples_per_symbol
        Oversampling factor (T_symbol / T_sample).
    roll_off
        Excess-bandwidth factor β ∈ [0, 1].
    """
    if not 0.0 <= roll_off <= 1.0:
        raise ValueError(f"roll_off must be in [0, 1], got {roll_off}")
    if samples_per_symbol < 1:
        raise ValueError(f"samples_per_symbol must be >= 1, got {samples_per_symbol}")

    sps = float(samples_per_symbol)
    beta = float(roll_off)
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / sps  # in symbol periods

    h = np.empty(n_taps, dtype=np.float64)
    # Vectorized, with explicit handling of two singularities.
    pi = np.pi
    # Limit at t = 0: h(0) = 1 + β(4/π - 1)
    zero_mask = np.isclose(t, 0.0, atol=1e-15)
    # Limit at t = ±1/(4β): h = (β/√2) * [(1 + 2/π) sin(π/(4β)) + (1 - 2/π) cos(π/(4β))]
    if beta > 0:
        sing = 1.0 / (4.0 * beta)
        sing_mask = np.isclose(np.abs(t), sing, atol=1e-15)
    else:
        sing_mask = np.zeros_like(t, dtype=bool)
    regular = ~(zero_mask | sing_mask)

    tt = t[regular]
    num = np.sin(pi * (1.0 - beta) * tt) + 4.0 * beta * tt * np.cos(pi * (1.0 + beta) * tt)
    denom = pi * tt * (1.0 - (4.0 * beta * tt) ** 2)
    h[regular] = num / denom

    h[zero_mask] = 1.0 + beta * (4.0 / pi - 1.0)
    if beta > 0:
        val = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / pi) * np.sin(pi / (4.0 * beta))
            + (1.0 - 2.0 / pi) * np.cos(pi / (4.0 * beta))
        )
        h[sing_mask] = val

    # Normalize to unit energy.
    h /= np.sqrt(np.sum(h**2))
    return h
