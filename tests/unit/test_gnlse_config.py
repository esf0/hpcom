"""Generalised-NLSE configuration: Fiber fields, validation, recentring, Raman models."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import Fiber
from hpcom.fiber import LossSpectrum
from hpcom.multicore import MulticoreFiber, linear_array_coupling
from hpcom.raman import RamanResponse
from hpcom.units import SPEED_OF_LIGHT_M_PER_S as C

SPECTRUM = LossSpectrum(wavelengths_m=[1.50e-6, 1.55e-6, 1.60e-6], alpha_db_per_km=[0.3, 0.2, 0.25])


def test_defaults_are_the_plain_nlse():
    f = Fiber(length_km=1.0)
    assert f.higher_order_betas == () and f.wavelength_m == 1550e-9
    assert f.loss_spectrum is None and f.raman is None and not f.self_steepening
    assert f.effective_shock_time_s == 0.0
    assert f.dispersion_betas == (0.0, f.beta2_s2_per_m, 0.0)
    assert Fiber.ssmf(1.0, wavelength_m=1310e-9).wavelength_m == 1310e-9


def test_gnlse_fiber_round_trips_through_json():
    f = Fiber(
        length_km=0.5,
        higher_order_betas=(1e-55, -2e-70),
        raman=RamanResponse(model="lin-agrawal", fraction=0.2),
        self_steepening=True,
        shock_time_s=0.6e-15,
        scheme="rk4ip",
    ).with_loss_spectrum(SPECTRUM)
    assert Fiber.model_validate_json(f.model_dump_json()) == f


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"raman": RamanResponse(), "polarization": "dual"}, "single polarisation"),
        ({"self_steepening": True}, "rk4ip"),
        ({"shock_time_s": 1e-15}, "needs self_steepening"),
        ({"higher_order_betas": (math.inf,)}, "finite"),
        ({"loss_spectrum": SPECTRUM, "alpha_db_per_km": 0.21}, "with_loss_spectrum"),
        ({"wavelength_m": 0.0}, "greater than 0"),
    ],
)
def test_invalid_gnlse_fibers_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        Fiber(length_km=1.0, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wavelengths_m": [1.5e-6], "alpha_db_per_km": [0.2]}, "at least 2"),
        ({"wavelengths_m": [1.6e-6, 1.5e-6], "alpha_db_per_km": [0.2, 0.2]}, "increasing"),
        ({"wavelengths_m": [1.5e-6, 1.6e-6], "alpha_db_per_km": [0.2, -0.1]}, ">= 0"),
        ({"wavelengths_m": [1.5e-6, 1.6e-6], "alpha_db_per_km": [0.2]}, "same length"),
    ],
)
def test_invalid_loss_spectra_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        LossSpectrum(**kwargs)


def test_loss_spectrum_interpolates_linearly_and_clamps():
    np.testing.assert_allclose(SPECTRUM.at([1.525e-6, 1.575e-6]), [0.25, 0.225])
    np.testing.assert_allclose(SPECTRUM.at([1.0e-6, 2.0e-6, np.inf]), [0.3, 0.25, 0.25])
    f = Fiber(length_km=1.0, wavelength_m=1.525e-6).with_loss_spectrum(SPECTRUM)
    assert f.alpha_db_per_km == pytest.approx(0.25)
    assert f.with_loss_spectrum(None).loss_spectrum is None


def test_shock_time_defaults_to_one_over_the_carrier_frequency():
    f = Fiber(length_km=1.0, self_steepening=True, scheme="rk4ip")
    assert f.effective_shock_time_s == pytest.approx(1550e-9 / (2 * math.pi * C))
    assert f.replace(shock_time_s=2e-15).effective_shock_time_s == 2e-15


BETAS = (3e-13, -2.2e-26, 1.3e-40, -3e-55, 2e-69, 7e-84)


def _taylor_fiber(**kwargs) -> Fiber:
    return Fiber(
        length_km=1.0,
        beta1_s_per_m=BETAS[0],
        beta2_s2_per_m=BETAS[1],
        beta3_s3_per_m=BETAS[2],
        higher_order_betas=BETAS[3:],
        **kwargs,
    )


def _d(betas: tuple[float, ...], omega: np.ndarray) -> np.ndarray:
    return sum(b * (-omega) ** n / math.factorial(n) for n, b in enumerate(betas, start=1))


def test_recentred_taylor_series_is_the_same_polynomial():
    """D(ω' + ω_k) = D'(ω') + D(ω_k) for every ω' (the phase is dispersion_phase_per_m)."""
    fiber = _taylor_fiber()
    f_k = -1.7e12
    moved = fiber.recentred(f_k)
    w_k = 2 * math.pi * f_k
    omega = 2 * math.pi * np.linspace(-3e12, 3e12, 7)
    np.testing.assert_allclose(
        _d(moved.dispersion_betas, omega) + fiber.dispersion_phase_per_m(f_k),
        _d(fiber.dispersion_betas, omega + w_k),
        rtol=1e-12,
    )
    assert fiber.dispersion_phase_per_m(f_k) == pytest.approx(_d(BETAS, np.array(w_k)), rel=1e-14)
    assert moved.wavelength_m == pytest.approx(C / (C / 1550e-9 - f_k))
    assert fiber.recentred(0.0) is fiber


def test_recentring_moves_loss_and_the_shock_factor():
    """With self-steepening γ'(1 − τ'ω') = γ(1 − τ(ω' + ω_k))."""
    f_k = 2e12
    w_k = 2 * math.pi * f_k
    for shock in (None, 3e-15):
        fiber = _taylor_fiber(self_steepening=True, shock_time_s=shock, scheme="rk4ip")
        fiber = fiber.with_loss_spectrum(SPECTRUM)
        moved = fiber.recentred(f_k)
        tau, tau_k = fiber.effective_shock_time_s, moved.effective_shock_time_s
        for omega in (-1e13, 0.0, 4e13):
            assert moved.gamma_per_w_per_km * (1 - tau_k * omega) == pytest.approx(
                fiber.gamma_per_w_per_km * (1 - tau * (omega + w_k)), rel=1e-12
            )
        assert moved.alpha_db_per_km == pytest.approx(float(SPECTRUM.at(moved.wavelength_m)))
    plain = _taylor_fiber().recentred(f_k)
    assert plain.gamma_per_w_per_km == _taylor_fiber().gamma_per_w_per_km


def test_recentred_multicore_fiber_moves_the_skew_into_core_offsets():
    fiber = MulticoreFiber.from_fiber(
        Fiber(length_km=1.0),
        linear_array_coupling(3, 1e-3),
        core_beta1_offsets_s_per_m=[0.0, 1e-12, -2e-12],
    )
    moved = fiber.recentred(5e9)
    assert isinstance(moved, MulticoreFiber)
    np.testing.assert_allclose(
        moved.core_beta0_offsets_per_m, -np.array([0.0, 1e-12, -2e-12]) * 2 * math.pi * 5e9
    )


# ---- Raman models ----------------------------------------------------------


@pytest.mark.parametrize("model", ["blow-wood", "lin-agrawal"])
def test_raman_transfer_function_is_the_transform_of_the_impulse_response(model):
    r = RamanResponse(model=model)
    dt = 0.01e-15
    t = np.arange(0, 3e-12, dt)
    h = r.impulse(t)
    for f in (0.0, 3e12, 13.2e12, 40e12):
        w = 2 * math.pi * f
        numeric = (np.sum(h * np.exp(-1j * w * t)) - 0.5 * h[0]) * dt  # trapezoid
        assert abs(numeric - r.transfer(w)) < 2e-7
    assert r.transfer(0.0) == pytest.approx(1.0, abs=1e-15)
    assert r.impulse(np.array([-1e-15]))[0] == 0.0
    assert r.response_time_s == pytest.approx(r.f_r * np.sum(t * h) * dt, rel=1e-6)


@pytest.mark.parametrize(
    ("model", "f_r", "peak_thz"),
    [("blow-wood", 0.18, 13.08), ("lin-agrawal", 0.245, 13.04)],
)
def test_raman_gain_peaks_near_13_thz_at_the_measured_level(model, f_r, peak_thz):
    """SSMF (γ = 1.27 /W/km): g_R/A_eff ≈ 0.6–0.8 /W/km co-polarised at ≈ 13 THz."""
    r = RamanResponse(model=model)
    assert r.f_r == f_r
    shift = np.linspace(0.05e12, 40e12, 8000)
    g = r.gain_per_w_per_m(1.27, shift)
    assert shift[np.argmax(g)] / 1e12 == pytest.approx(peak_thz, abs=0.02)
    assert 0.6e-3 < g.max() < 0.8e-3
    assert np.all(r.gain_per_w_per_m(1.27, -shift) < 0)  # anti-Stokes side is absorbed
    assert RamanResponse(model=model, fraction=0.1).f_r == 0.1
