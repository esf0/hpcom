"""T11.5: AWGN shaping gain.

Compares the SNR required to operate at a fixed mutual-information target
``R*`` (bits/symbol) over an AWGN channel with a 64-point QAM constellation,
once driven by a uniform symbol distribution and once driven by a
Maxwell–Boltzmann shaped distribution. Both constellations are normalised
to unit *expected* transmitted power, so the comparison is at matched
average TX power and matched information rate.

The theoretical asymptotic shaping gain over a square QAM ranges from ~0.3 dB
at MI just below ``log₂M`` to ~1.53 dB as MI → channel-capacity limit. The
acceptance threshold for this test (≥ 0.5 dB at ``R* = 4 bits/symbol`` over
64-QAM) is comfortably inside that window.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import (
    ConstellationConfig,
    maxwell_boltzmann,
    square_qam,
)

# ---------------------------------------------------------------------------
# MI estimator
# ---------------------------------------------------------------------------

def _mutual_information_awgn(
    constellation: np.ndarray,
    pmf: np.ndarray,
    snr_db: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Monte-Carlo estimate of ``I(X; Y)`` in bits over a complex AWGN channel.

    ``constellation`` carries the symbol coordinates. ``pmf`` is the symbol
    distribution (uniform or shaped). ``snr_db = 10·log10(E[|X|²] / N₀)``;
    here ``E[|X|²] = Σ p_i |s_i|²``. Complex AWGN with total variance ``N₀``
    is split equally between real and imag axes.
    """
    es = float(np.dot(pmf, np.abs(constellation) ** 2))
    n0 = es / (10.0 ** (snr_db / 10.0))

    idx = rng.choice(constellation.size, size=n_samples, p=pmf)
    x = constellation[idx]
    noise = np.sqrt(n0 / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    y = x + noise

    # log p(y | x_i) up to the y-independent normalisation 1/(π N₀), which
    # cancels between log p(y|x_tx) and log Σ p_i p(y|x_i).
    log_pyx_grid = -(np.abs(y[:, None] - constellation[None, :]) ** 2) / n0
    # log Σ_i p_i exp(log_pyx_grid)  via log-sum-exp.
    m = log_pyx_grid.max(axis=1, keepdims=True)
    log_py = m.squeeze(axis=1) + np.log(
        np.sum(pmf[None, :] * np.exp(log_pyx_grid - m), axis=1)
    )

    log_pyx_tx = log_pyx_grid[np.arange(n_samples), idx]
    mi_nats = float(np.mean(log_pyx_tx - log_py))
    return mi_nats / np.log(2.0)


def _snr_at_target_mi(
    constellation: np.ndarray,
    pmf: np.ndarray,
    target_mi: float,
    snr_grid_db: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Linear-interpolate the SNR at which estimated MI crosses ``target_mi``."""
    mis = np.array([
        _mutual_information_awgn(constellation, pmf, s, n_samples, rng)
        for s in snr_grid_db
    ])
    if not (mis[0] < target_mi < mis[-1]):
        raise AssertionError(
            f"target_mi {target_mi} not bracketed by MI grid "
            f"[{mis[0]:.3f}, {mis[-1]:.3f}] (snr_grid_db={snr_grid_db})"
        )
    return float(np.interp(target_mi, mis, snr_grid_db))


# ---------------------------------------------------------------------------
# Acceptance test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_64qam_maxwell_boltzmann_shaping_gain_at_4_bits():
    """64-QAM with M-B PMF beats uniform 64-QAM by ≥ 0.5 dB at MI = 4 bits."""
    rng = np.random.default_rng(2026)
    target_mi = 4.0

    points_raw, _ = square_qam(64)
    # Normalise both systems to unit *expected* power under their PMF
    # (this is what ConstellationConfig does):
    uniform_pmf = np.full(64, 1.0 / 64)
    cfg_uniform = ConstellationConfig.from_points(points_raw, probabilities=uniform_pmf)

    # Maxwell-Boltzmann ν tuning: shaping gain at MI target R* is largest
    # when H(p) sits above R* but the PMF is meaningfully non-uniform. ν=1.2
    # over unit-power 64-QAM gives H(p) ≈ 5.7 bits and roughly 0.7 dB gain
    # at MI = 4 — comfortably above the 0.5 dB acceptance threshold.
    nu = 1.2
    shaped_pmf = maxwell_boltzmann(points_raw, nu=nu)
    cfg_shaped = ConstellationConfig.from_points(
        points_raw, probabilities=shaped_pmf
    )

    # H(p) sanity-check: must exceed target_mi so MI can actually reach it.
    h_shaped = float(-np.sum(shaped_pmf * np.log2(shaped_pmf + 1e-300)))
    assert h_shaped > target_mi + 0.5, (
        f"Maxwell-Boltzmann entropy {h_shaped:.2f} bits is too low to "
        f"support MI = {target_mi} bits — pick a smaller ν"
    )

    # Sweep SNR in a window that brackets MI = 4 bits for both systems.
    snr_grid = np.arange(6.0, 18.5, 0.5)
    n_samples = 20_000

    snr_uniform = _snr_at_target_mi(
        cfg_uniform.points, cfg_uniform.probabilities, target_mi,
        snr_grid, n_samples, rng,
    )
    snr_shaped = _snr_at_target_mi(
        cfg_shaped.points, cfg_shaped.probabilities, target_mi,
        snr_grid, n_samples, rng,
    )
    gain_db = snr_uniform - snr_shaped

    # Report for the user checkpoint.
    print(
        f"\n[shaping-gain] target MI = {target_mi} bits/sym\n"
        f"  uniform 64-QAM:   SNR = {snr_uniform:.3f} dB\n"
        f"  M-B (ν={nu}): SNR = {snr_shaped:.3f} dB  (H(p) = {h_shaped:.2f} bits)\n"
        f"  shaping gain = {gain_db:.3f} dB"
    )
    assert gain_db >= 0.5, (
        f"shaping gain {gain_db:.3f} dB < 0.5 dB acceptance threshold"
    )
