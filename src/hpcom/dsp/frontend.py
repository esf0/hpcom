"""Receiver front end: ideal band-limiting and exact resampling of periodic fields.

Simulated fields are periodic (pulse shaping and split-step propagation are
circular), so both operations are done exactly in the frequency domain.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray

__all__ = ["lowpass", "resample"]


def lowpass(field: ComplexArray, dt_s: float, bandwidth_hz: float) -> npt.NDArray[np.complex128]:
    """Ideal (brick-wall) low-pass along the last axis, keeping ``|f| ≤ bandwidth_hz / 2``.

    ``bandwidth_hz`` is the two-sided (optical) bandwidth of the complex
    baseband signal, e.g. ``(1 + β)·R_s`` for one RRC channel.
    """
    if not bandwidth_hz > 0:
        raise ValueError(f"bandwidth_hz must be positive, got {bandwidth_hz}")
    x = np.asarray(field, dtype=np.complex128)
    f = np.fft.fftfreq(x.shape[-1], d=dt_s)
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[..., np.abs(f) > bandwidth_hz / 2.0] = 0.0
    out: npt.NDArray[np.complex128] = np.fft.ifft(spectrum, axis=-1)
    return out


def resample(field: ComplexArray, n_samples: int) -> npt.NDArray[np.complex128]:
    """Band-limited resampling of a periodic signal to ``n_samples`` over the same window.

    Along the last axis. Every frequency below the Nyquist frequency of the
    coarser grid is kept exactly; everything above is removed (ideal
    anti-aliasing when downsampling, ideal interpolation when upsampling).
    The Nyquist bin of an even-length coarser grid — whose sign is ambiguous
    — is dropped; it carries no energy for signals band-limited below it.

    For such signals the output samples are the exact band-limited
    interpolation of the input at ``t = k·T/n_samples``, and resampling back
    returns the input to rounding error.
    """
    x: npt.NDArray[np.complex128] = np.asarray(field, dtype=np.complex128)
    n_in = x.shape[-1]
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if n_samples == n_in:
        same: npt.NDArray[np.complex128] = x.copy()
        return same
    m = min(n_in, n_samples)
    k = (
        np.fft.fftfreq(m, d=1.0 / m).round().astype(np.intp)
    )  # integer frequencies of the coarse grid
    k = k[np.abs(2 * k) < m]
    spectrum = np.fft.fft(x, axis=-1)
    out_spectrum = np.zeros((*x.shape[:-1], n_samples), dtype=np.complex128)
    out_spectrum[..., k % n_samples] = spectrum[..., k % n_in]
    out: npt.NDArray[np.complex128] = np.fft.ifft(out_spectrum, axis=-1) * (n_samples / n_in)
    return out
