"""Amplifier ASE statistics (Test F).

The existing `test_amplifier.py` pins variance and dual-pol independence; this
file targets second-order properties of the noise:

- ⟨Re·Im⟩ ≈ 0 (circularly-symmetric complex Gaussian)
- sample-wise excess kurtosis ≈ 0 (Gaussian, not e.g. Laplace or uniform)
- per-sample temporal independence (autocorrelation ≈ δ)

Catches a class of regression where the noise generator silently switches to
the wrong distribution or accidentally introduces serial correlation.
"""

from __future__ import annotations

import numpy as np

from hpcom.amplifier import EDFA


def _ase_only_samples(n: int, *, seed: int = 0) -> np.ndarray:
    """Run an EDFA on a zero field so the output is pure ASE noise."""
    edfa = EDFA(gain_db=20.0, noise_figure_db=5.0, wavelength_m=1550e-9)
    zero_field = np.zeros(n, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    return edfa.amplify(zero_field, dt_s=1.0 / 80e9, rng=rng)


def test_ase_quadratures_are_uncorrelated():
    """E[Re · Im] ≈ 0 — required by circular complex Gaussianity."""
    n = 200_000
    noise = _ase_only_samples(n)
    re = noise.real
    im = noise.imag
    var = float(np.var(re))
    cross = float(np.mean(re * im))
    # Cross-correlation should be O(sqrt(var² / n)). 5 σ tolerance.
    bound = 5.0 * var / np.sqrt(n)
    assert abs(cross) < bound, f"|E[Re·Im]| = {abs(cross):.3e} exceeds {bound:.3e}"


def test_ase_kurtosis_is_close_to_zero():
    """Excess kurtosis of each quadrature should be ≈ 0 for Gaussian noise.
    Uniform noise → -1.2; Laplace → 3.0 — a wrong sampler would fail loudly."""
    n = 100_000
    noise = _ase_only_samples(n, seed=1)
    for arr, label in [(noise.real, "Re"), (noise.imag, "Im")]:
        m = arr - arr.mean()
        var = float(np.mean(m**2))
        kurt = float(np.mean(m**4)) / (var**2) - 3.0
        # Std error of kurtosis estimator ~ sqrt(24/n); 5σ tolerance.
        bound = 5.0 * np.sqrt(24.0 / n)
        assert abs(kurt) < bound, f"{label} excess kurtosis {kurt:.3f} exceeds {bound:.3f}"


def test_ase_samples_are_temporally_independent():
    """Lag-1 sample autocorrelation should be ≈ 0 (white noise)."""
    n = 50_000
    noise = _ase_only_samples(n, seed=2)
    var = float(np.var(noise))
    # Complex autocorrelation at lag 1, normalized by variance.
    autocorr_1 = float(np.real(np.mean(noise[1:] * np.conj(noise[:-1])))) / var
    # 5σ tolerance: sample autocorrelation std error ~ 1/√n for white noise.
    bound = 5.0 / np.sqrt(n)
    assert abs(autocorr_1) < bound, (
        f"lag-1 autocorr {autocorr_1:.4f} exceeds white-noise tolerance {bound:.4f}"
    )
