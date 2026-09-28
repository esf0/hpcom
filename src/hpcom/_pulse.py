r"""Root-raised-cosine pulse shaping and matched filtering.

Three implementations, selected by ``span_symbols`` and ``convolution``:

============================  ==============================================
``span_symbols=None``         Exact RRC, applied in the frequency domain
(``convolution="circular"``)  (periodic signal). Zero ISI to rounding error
                              and strictly band-limited to (1+β)R_s/2.
``span_symbols=S``,           Truncated FIR of ``S·sps + 1`` taps applied as
``convolution="circular"``    a circular convolution (periodic signal): the
                              filter of a real transmitter, without block
                              edges. Truncation ISI decays with ``S``.
``span_symbols=S``,           Same FIR, linear convolution of a finite burst
``convolution="linear"``      ("same"-length output). The first/last ≈ ``S``
                              symbols carry edge transients.
============================  ==============================================

Normalisation (all modes). With symbols :math:`a_k` placed every ``sps``
samples, the transmit filter is scaled so that the output has mean power
:math:`E|a|^2` (for i.i.d. symbols), and the receive filter so that the
noiseless cascade returns :math:`a_k` at the symbol instants with unit gain.
White noise of per-sample variance :math:`\sigma^2` leaves the receive filter
with variance :math:`\sigma^2/\text{sps}` per symbol, so a field of power
:math:`P` in noise of PSD :math:`N_0` gives :math:`\text{SNR} = P/(N_0 R_s)`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Convolution = Literal["circular", "linear"]


def root_raised_cosine(
    n_taps: int, samples_per_symbol: int, roll_off: float
) -> npt.NDArray[np.float64]:
    """Time-domain RRC FIR of ``n_taps`` taps, centred, with unit energy (Σh² = 1).

    ``n_taps`` should be odd so the peak falls on a sample.
    """
    if not 0.0 <= roll_off <= 1.0:
        raise ValueError(f"roll_off must be in [0, 1], got {roll_off}")
    if samples_per_symbol < 1:
        raise ValueError(f"samples_per_symbol must be >= 1, got {samples_per_symbol}")
    if n_taps < 1:
        raise ValueError(f"n_taps must be >= 1, got {n_taps}")
    beta = float(roll_off)
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / samples_per_symbol  # in symbol periods
    h = np.empty(n_taps, dtype=np.float64)
    at_zero = np.isclose(t, 0.0, atol=1e-12)
    at_sing = (
        np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-12)
        if beta > 0
        else np.zeros_like(at_zero)
    )
    reg = ~(at_zero | at_sing)
    tt = t[reg]
    h[reg] = (
        np.sin(np.pi * (1.0 - beta) * tt) + 4.0 * beta * tt * np.cos(np.pi * (1.0 + beta) * tt)
    ) / (np.pi * tt * (1.0 - (4.0 * beta * tt) ** 2))
    h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    if beta > 0:
        h[at_sing] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
    h /= np.sqrt(np.sum(h**2))
    return h


def raised_cosine_spectrum(f_over_rs: npt.ArrayLike, roll_off: float) -> npt.NDArray[np.float64]:
    """Raised-cosine spectrum with unit peak; frequency in units of the symbol rate.

    Satisfies the Nyquist folding identity ``RC(f) + RC(f − 1) = 1`` for
    ``f ∈ [0, 1]`` (for β = 0 the band edge ``|f| = 1/2`` takes the value ½).
    """
    beta = float(roll_off)
    af = np.abs(np.asarray(f_over_rs, dtype=np.float64))
    lo, hi = (1.0 - beta) / 2.0, (1.0 + beta) / 2.0
    out = np.zeros_like(af)
    out[af < lo] = 1.0
    if beta > 0:
        band = (af >= lo) & (af <= hi)
        out[band] = 0.5 * (1.0 + np.cos(np.pi / beta * (af[band] - lo)))
    else:
        out[np.isclose(af, 0.5, rtol=0, atol=1e-12)] = 0.5
    return out


def fir_taps(
    samples_per_symbol: int, roll_off: float, span_symbols: int
) -> npt.NDArray[np.float64]:
    """Unit-energy RRC FIR spanning ``span_symbols`` symbols (``span·sps + 1`` taps)."""
    return root_raised_cosine(span_symbols * samples_per_symbol + 1, samples_per_symbol, roll_off)


def _circular_responses(
    n_samples: int, samples_per_symbol: int, roll_off: float, span_symbols: int | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(H_tx, H_rx)`` on the length-``n_samples`` DFT grid (real, zero-phase)."""
    sps = samples_per_symbol
    if span_symbols is None:
        rc = raised_cosine_spectrum(np.fft.fftfreq(n_samples, d=1.0 / sps), roll_off)
        # H_tx·H_rx = sps·RC → unit-gain Nyquist cascade; mean|H_tx|² = sps
        # because the DFT-grid mean of RC is exactly 1/sps (folding identity).
        return sps * np.sqrt(rc), np.sqrt(rc)
    taps = fir_taps(sps, roll_off, span_symbols)
    if taps.size > n_samples:
        raise ValueError(
            f"pulse_span_symbols={span_symbols} needs {taps.size} taps, more than the "
            f"{n_samples} samples of the signal; use fewer span symbols or more symbols"
        )
    half = (taps.size - 1) // 2
    circ = np.zeros(n_samples)
    circ[: half + 1] = taps[half:]
    circ[n_samples - half :] = taps[:half]
    h = np.fft.fft(circ).real  # symmetric taps → real response
    return np.sqrt(sps) * h, h / np.sqrt(sps)


def pulse_shape(
    symbols: npt.ArrayLike,
    samples_per_symbol: int,
    roll_off: float,
    span_symbols: int | None,
    convolution: Convolution,
) -> npt.NDArray[np.complex128]:
    """Upsample and RRC-filter ``symbols`` along the last axis → ``n_symbols·sps`` samples."""
    a = np.asarray(symbols, dtype=np.complex128)
    sps = samples_per_symbol
    n = a.shape[-1] * sps
    up = np.zeros((*a.shape[:-1], n), dtype=np.complex128)
    up[..., ::sps] = a
    if convolution == "circular":
        h_tx, _ = _circular_responses(n, sps, roll_off, span_symbols)
        out: npt.NDArray[np.complex128] = np.fft.ifft(np.fft.fft(up, axis=-1) * h_tx, axis=-1)
        return out
    if span_symbols is None:
        raise ValueError("linear convolution needs a finite pulse_span_symbols")
    taps = np.sqrt(sps) * fir_taps(sps, roll_off, span_symbols)
    return _linear_same(up, taps)


def matched_filter(
    field: npt.ArrayLike,
    samples_per_symbol: int,
    roll_off: float,
    span_symbols: int | None,
    convolution: Convolution,
) -> npt.NDArray[np.complex128]:
    """RRC matched filter + symbol-rate sampling along the last axis (inverse of :func:`pulse_shape`)."""
    x = np.asarray(field, dtype=np.complex128)
    sps = samples_per_symbol
    n = x.shape[-1]
    if n % sps:
        raise ValueError(f"field length {n} is not a multiple of samples_per_symbol={sps}")
    if convolution == "circular":
        _, h_rx = _circular_responses(n, sps, roll_off, span_symbols)
        y = np.fft.ifft(np.fft.fft(x, axis=-1) * h_rx, axis=-1)
    else:
        if span_symbols is None:
            raise ValueError("linear convolution needs a finite pulse_span_symbols")
        y = _linear_same(x, fir_taps(sps, roll_off, span_symbols) / np.sqrt(sps))
    out: npt.NDArray[np.complex128] = y[..., ::sps]
    return out


def _linear_same(
    x: npt.NDArray[np.complex128], taps: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex128]:
    """Linear convolution along the last axis, trimmed to the input length (centred taps)."""
    delay = (taps.size - 1) // 2
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        out[i] = np.convolve(row, taps)[delay : delay + n]
    return out.reshape(x.shape)
