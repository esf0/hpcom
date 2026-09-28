"""Receiver front end: ideal low-pass and exact band-limited resampling."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.dsp import lowpass, resample


def _bandlimited(n: int, max_bin: int, shape: tuple[int, ...] = (), seed: int = 0) -> np.ndarray:
    """Periodic complex signal with spectral content only at |k| ≤ max_bin."""
    rng = np.random.default_rng(seed)
    spectrum = np.zeros((*shape, n), dtype=complex)
    k = np.arange(-max_bin, max_bin + 1)
    spectrum[..., k % n] = rng.standard_normal((*shape, k.size)) + 1j * rng.standard_normal(
        (*shape, k.size)
    )
    return np.fft.ifft(spectrum, axis=-1)


@pytest.mark.parametrize(("n", "factor"), [(64, 2), (64, 4), (63, 3), (100, 2)])
def test_upsampling_keeps_the_original_samples_and_downsampling_restores_them(n, factor):
    x = _bandlimited(n, n // 2 - 1)
    up = resample(x, n * factor)
    np.testing.assert_allclose(up[::factor], x, atol=1e-13)
    np.testing.assert_allclose(resample(up, n), x, atol=1e-13)


@pytest.mark.parametrize("n_out", [32, 33, 48, 100])
def test_downsampling_a_signal_below_the_new_nyquist_is_lossless(n_out):
    x = _bandlimited(128, 15, shape=(2, 3))
    down = resample(x, n_out)
    assert down.shape == (2, 3, n_out)
    np.testing.assert_allclose(resample(down, 128), x, atol=1e-13)


def test_downsampling_removes_everything_above_the_new_nyquist():
    n = 256
    t = np.arange(n)
    inside, outside = np.exp(2j * np.pi * 10 * t / n), np.exp(2j * np.pi * 40 * t / n)
    np.testing.assert_allclose(resample(inside + outside, 64), resample(inside, 64), atol=1e-13)


def test_resample_to_the_same_length_is_a_copy_and_bad_lengths_are_rejected():
    x = _bandlimited(16, 4)
    out = resample(x, 16)
    np.testing.assert_array_equal(out, x)
    assert out is not x
    with pytest.raises(ValueError, match="n_samples"):
        resample(x, 0)


def test_lowpass_keeps_the_band_and_removes_the_rest():
    n, dt = 400, 1e-12  # 2.5 GHz bins
    t = np.arange(n) * dt
    tone = lambda f: np.exp(2j * np.pi * f * t)  # noqa: E731
    x = tone(10e9) + tone(-20e9) + tone(30e9) + tone(-60e9)
    np.testing.assert_allclose(lowpass(x, dt, 2 * 30e9), x - tone(-60e9), atol=1e-12)
    np.testing.assert_allclose(
        lowpass(x[None].repeat(2, 0), dt, 50e9)[1], tone(10e9) + tone(-20e9), atol=1e-12
    )
    with pytest.raises(ValueError, match="positive"):
        lowpass(x, dt, 0.0)
