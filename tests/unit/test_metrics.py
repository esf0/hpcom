"""Metric functions: BER, EVM, SNR, Q-factor, MI, GMI."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.metrics import (
    Metrics,
    bit_error_rate,
    evm_rms,
    gmi_gaussian,
    mi_gaussian,
    q_factor_db,
    snr_db,
)
from hpcom.modulation import ConstellationConfig, maxwell_boltzmann_for_entropy


def _bpsk_capacity(amplitude: float, sigma_dim: float) -> float:
    """C = 1 − E_n[log₂(1 + e^{−2A(A+n)/σ²})], n ~ N(0, σ²) (Gauss–Hermite, 120 nodes)."""
    x, w = np.polynomial.hermite.hermgauss(120)
    n = np.sqrt(2.0) * sigma_dim * x
    return float(
        1.0
        - np.sum(w * np.log2(1.0 + np.exp(-2 * amplitude * (amplitude + n) / sigma_dim**2)))
        / np.sqrt(np.pi)
    )


def _awgn(cfg: ConstellationConfig, n: int, sigma2: float, seed: int = 0, pmf=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(cfg.modulation_order, size=n, p=pmf)
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return cfg.points[idx] + noise, idx


def test_bit_error_rate_counts():
    ber, errors, n = bit_error_rate(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))
    assert (ber, errors, n) == (0.5, 2, 4)
    assert bit_error_rate(np.array([], np.uint8), np.array([], np.uint8)) == (0.0, 0, 0)
    with pytest.raises(ValueError, match="shape mismatch"):
        bit_error_rate(np.zeros(3), np.zeros(4))


def test_evm_and_snr_are_consistent():
    x = np.array([1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j]) / np.sqrt(2)
    y = x + 0.1
    assert evm_rms(y, x) == pytest.approx(0.1)
    assert snr_db(y, x) == pytest.approx(20.0)
    assert evm_rms(x, x) == 0.0 and snr_db(x, x) == float("inf")
    assert evm_rms(np.ones(2), np.zeros(2)) == float("inf")
    with pytest.raises(ValueError, match="shape mismatch"):
        evm_rms(np.ones(2), np.ones(3))


@pytest.mark.parametrize("ber", [1e-9, 1e-3, 3.8e-3, 2e-2, 0.2])
def test_q_factor_from_ber(ber):
    """BER = Q(q) with Q the Gaussian tail ⇒ q = Q⁻¹(BER) (independent of erfcinv)."""
    from scipy.stats import norm

    assert q_factor_db(ber) == pytest.approx(20 * np.log10(norm.isf(ber)), abs=1e-9)
    assert q_factor_db(1e-3) == pytest.approx(9.7998, abs=1e-4)  # Q = 3.0902


@pytest.mark.parametrize("ber", [0.0, 0.5, 0.7, -1.0])
def test_q_factor_undefined_outside_the_open_interval(ber):
    assert q_factor_db(ber) is None


@pytest.mark.parametrize("snr_db_value", [0.0, 5.0, 10.0])
def test_qpsk_mi_and_gmi_equal_twice_the_bpsk_capacity(snr_db_value):
    cfg = ConstellationConfig.from_square_qam(4)
    sigma2 = 10 ** (-snr_db_value / 10)
    y, idx = _awgn(cfg, 200_000, sigma2)
    expected = 2 * _bpsk_capacity(1 / np.sqrt(2), np.sqrt(sigma2 / 2))
    assert mi_gaussian(y, idx, cfg.points) == pytest.approx(expected, abs=0.01)
    assert gmi_gaussian(y, idx, cfg.points, cfg.bit_labels) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("order", [16, 64])
def test_rates_approach_the_entropy_without_noise_and_zero_in_heavy_noise(order):
    cfg = ConstellationConfig.from_square_qam(order)
    k = np.log2(order)
    y, idx = _awgn(cfg, 20_000, 1e-6)
    assert mi_gaussian(y, idx, cfg.points) == pytest.approx(k, abs=1e-6)
    assert gmi_gaussian(y, idx, cfg.points, cfg.bit_labels) == pytest.approx(k, abs=1e-6)
    y, idx = _awgn(cfg, 20_000, 1e3)
    assert mi_gaussian(y, idx, cfg.points) < 0.01
    assert gmi_gaussian(y, idx, cfg.points, cfg.bit_labels) < 0.01


@pytest.mark.parametrize("snr_db_value", [4.0, 10.0, 16.0])
def test_gmi_never_exceeds_mi(snr_db_value):
    cfg = ConstellationConfig.from_square_qam(64)
    y, idx = _awgn(cfg, 50_000, 10 ** (-snr_db_value / 10), seed=3)
    assert (
        gmi_gaussian(y, idx, cfg.points, cfg.bit_labels) <= mi_gaussian(y, idx, cfg.points) + 1e-9
    )


def test_rates_with_a_shaped_pmf_are_bounded_by_its_entropy():
    base = ConstellationConfig.from_square_qam(64)
    pmf = maxwell_boltzmann_for_entropy(base.points, 5.0)
    cfg = base.with_probabilities(pmf)
    y, idx = _awgn(cfg, 50_000, 1e-6, pmf=pmf)
    assert mi_gaussian(y, idx, cfg.points, pmf) == pytest.approx(5.0, abs=0.02)
    assert gmi_gaussian(y, idx, cfg.points, cfg.bit_labels, pmf) == pytest.approx(5.0, abs=1e-6)


def test_long_streams_are_processed_in_chunks(monkeypatch):
    import hpcom.metrics as metrics

    cfg = ConstellationConfig.from_square_qam(16)
    y, idx = _awgn(cfg, 3001, 0.05)
    full = (mi_gaussian(y, idx, cfg.points), gmi_gaussian(y, idx, cfg.points, cfg.bit_labels))
    monkeypatch.setattr(metrics, "_CHUNK", 16 * 7)
    chunked = (mi_gaussian(y, idx, cfg.points), gmi_gaussian(y, idx, cfg.points, cfg.bit_labels))
    np.testing.assert_allclose(chunked, full, rtol=1e-12)


def test_metrics_model_serialises():
    m = Metrics(
        n_bits=10,
        n_bit_errors=1,
        ber=0.1,
        n_symbols=5,
        n_symbol_errors=1,
        ser=0.2,
        evm_rms=0.1,
        snr_db=20.0,
        q_factor_db=None,
        mi_bits=1.5,
        gmi_bits=1.4,
    )
    assert Metrics.model_validate_json(m.model_dump_json()) == m
