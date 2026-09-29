"""Generalised-NLSE terms on every backend, against exact solutions.

Higher-order dispersion, frequency-dependent loss, the Raman response and
self-steepening (``docs/physics.rst``, *Generalised NLSE*):

* the dispersion moment theorem for β₁…β₆ (exact for any linear operator);
* each tone of a comb attenuated by the loss spectrum at its own wavelength;
* the linearised GNLSE around a CW pump — Raman gain, modulation
  instability, dispersion of every order and the shock factor together —
  against its closed-form 2×2 matrix exponential;
* self-steepening without dispersion against the exact solution by
  characteristics;
* photon-number conservation with Raman and self-steepening;
* exact invertibility of the split-step schemes with Raman, and agreement
  of every backend with the NumPy reference.

Fibers are duck-typed (:class:`StubFiber`), so these tests pin the backend
contract, not :class:`hpcom.Fiber`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from hpcom.backends import get_propagator
from hpcom.backends.base import kerr_power_response, linear_operator_exponent, shock_factor
from hpcom.fiber import LossSpectrum
from hpcom.raman import RamanResponse
from hpcom.units import SPEED_OF_LIGHT_M_PER_S
from tests.backends.test_propagator_conformance import StubFiber, StubScheme, _random_field

C = SPEED_OF_LIGHT_M_PER_S
RAMAN_MODELS = [pytest.param(RamanResponse(model=m), id=m) for m in ("blow-wood", "lin-agrawal")]


def _moments(a: np.ndarray, dt_s: float) -> tuple[float, float]:
    """Time centroid and variance of |A|² (pulse centred in the window)."""
    n = a.size
    t = (np.arange(n) - n // 2) * dt_s
    p = np.abs(np.fft.fftshift(a)) ** 2
    centroid = float(np.sum(t * p) / np.sum(p))
    return centroid, float(np.sum((t - centroid) ** 2 * p) / np.sum(p))


# ---------------------------------------------------------------------------
# Linear terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_dispersion_of_every_order_obeys_the_moment_theorem(any_backend, sign):
    r"""⟨t⟩(z) = ⟨t⟩₀ − z⟨D'⟩, ⟨t²⟩(z) = ⟨t²⟩₀ + z²⟨D'²⟩ for an unchirped pulse.

    With :math:`\tilde A(z,\omega) = \tilde A_0 e^{iD(\omega)z}`, :math:`t \leftrightarrow
    i\,d/d\omega`; the averages are over :math:`|\tilde A_0|^2`. Odd β's move
    the centroid, even ones spread the pulse; each β_n is chosen to
    contribute comparably at the pulse bandwidth.
    """
    n, dt, t0, length_m = 2**14, 40e-15, 300e-15, 10.0
    t = (np.arange(n) - n // 2) * dt
    a0 = np.fft.ifftshift(np.exp(-(t**2) / (2 * t0**2))).astype(complex)
    # β_k·ω_rms^{k−1}/(k−1)!·z = ±d_k·t₀: a spectral component at ω_rms is delayed by d_k·t₀.
    # (Small d₅, d₆ keep the far spectral tail, delayed ∝ ω⁴, ω⁵, inside the window.)
    w_rms = 1 / (math.sqrt(2) * t0)
    delays = (0.5, 0.5, 0.5, 0.3, 0.05, 0.01)
    betas = [
        sign * (-1) ** k * d * t0 * math.factorial(k - 1) / (length_m * w_rms ** (k - 1))
        for k, d in enumerate(delays, start=1)
    ]
    fiber = StubFiber(
        length_km=length_m / 1e3,
        beta1_s_per_m=betas[0],
        beta2_s2_per_m=betas[1],
        beta3_s3_per_m=betas[2],
        higher_order_betas=tuple(betas[3:]),
        gamma_per_w_per_km=0.0,
        n_steps=3,
    )
    z = fiber.length_km * 1e3
    out = get_propagator(any_backend).propagate(a0, fiber, dt_s=dt)
    omega = 2 * np.pi * np.fft.fftfreq(n, dt)
    d_prime = sum(
        b * (-1) ** k * k * omega ** (k - 1) / math.factorial(k) for k, b in enumerate(betas, 1)
    )
    weight = np.abs(np.fft.fft(a0)) ** 2
    mean = float(np.sum(weight * d_prime) / np.sum(weight))
    spread = float(np.sum(weight * (d_prime - mean) ** 2) / np.sum(weight))
    c0, v0 = _moments(a0, dt)
    c1, v1 = _moments(out, dt)
    assert abs(z * mean) > 0.2 * t0 and z**2 * spread > 0.2 * v0  # a non-trivial test
    assert c1 == pytest.approx(c0 - z * mean, abs=1e-9 * t0)
    assert v1 == pytest.approx(v0 + z**2 * spread, rel=1e-9)


def test_loss_spectrum_attenuates_each_tone_at_its_own_wavelength(any_backend):
    """Tone at baseband f is at λ = c/(ν₀ − f): +f is the longer wavelength."""
    n, df = 64, 250e9
    dt = 1 / (n * df)
    bins = np.array([-12, -5, 0, 3, 9, 14])
    t = np.arange(n) * dt
    a0 = sum(np.exp(2j * np.pi * k * df * t) for k in bins)
    spectrum = LossSpectrum(
        wavelengths_m=[1500e-9, 1550e-9, 1600e-9], alpha_db_per_km=[0.5, 0.2, 0.9]
    )
    fiber = StubFiber(
        length_km=20.0,
        alpha_db_per_km=0.2,
        beta2_s2_per_m=-2e-26,
        gamma_per_w_per_km=0.0,
        n_steps=4,
        loss_spectrum=spectrum,
    )
    out = get_propagator(any_backend).propagate(a0, fiber, dt_s=dt)
    power = np.abs(np.fft.fft(out) / n)[bins % n] ** 2
    wavelength = C / (C / 1550e-9 - bins * df)
    expected = 10 ** (-spectrum.at(wavelength) * fiber.length_km / 10)
    assert np.all(np.diff(wavelength) > 0)
    np.testing.assert_allclose(power, expected, rtol=1e-11)


# ---------------------------------------------------------------------------
# Linearised GNLSE around a CW pump: Raman gain, MI, dispersion, shock
# ---------------------------------------------------------------------------

_N_MI, _DF_MI, _P_MI, _BIN_MI = 64, 1e12, 2.0, 13  # seeds at ±13 THz: near the Raman peak


def _mi_fiber(shock: bool, method: str, n_steps: int, raman: RamanResponse) -> StubFiber:
    return StubFiber(
        length_km=0.5,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=4e-30,
        beta3_s3_per_m=-2e-43,
        higher_order_betas=(-5e-57, 2e-70),
        gamma_per_w_per_km=10.0,
        n_steps=n_steps,
        scheme=StubScheme(method=method),
        raman=raman,
        effective_shock_time_s=5e-15 if shock else 0.0,
    )


def _linearised_sidebands(fiber: StubFiber, x0: complex, y0: complex) -> np.ndarray:
    r"""Exact (x, y)(L) for x = ũ(+Ω), y = ũ(−Ω)* of A = (√P + u)e^{iγPz}.

    Linearising the GNLSE with Kerr response R̃ and shock factor S:

    .. math::

        \frac{d}{dz}\begin{pmatrix}x\\y\end{pmatrix} =
        \begin{pmatrix}
        iD_+ + i\gamma P(S_+(1+\tilde R) - 1) & i\gamma P S_+\tilde R\\
        -i\gamma P S_-\tilde R & -iD_- - i\gamma P(S_-(1+\tilde R) - 1)
        \end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix},

    :math:`D_\pm = D(\pm\Omega)`, :math:`S_\pm = S(\pm\Omega)`,
    :math:`\tilde R = \tilde R(\Omega)`.
    """
    n, dt = _N_MI, 1 / (_N_MI * _DF_MI)
    ip, im = _BIN_MI, _N_MI - _BIN_MI
    d = linear_operator_exponent(n, dt, fiber) / 1j
    r = kerr_power_response(n, dt, fiber)
    s = shock_factor(n, dt, fiber)
    s = np.ones(n) if s is None else s
    gp = fiber.gamma_per_w_per_km * 1e-3 * _P_MI
    m = np.array(
        [
            [1j * d[ip] + 1j * gp * (s[ip] * (1 + r[ip]) - 1), 1j * gp * s[ip] * r[ip]],
            [-1j * gp * s[im] * r[ip], -1j * d[im] - 1j * gp * (s[im] * (1 + r[ip]) - 1)],
        ]
    )
    return expm(m * fiber.length_km * 1e3) @ np.array([x0, y0])


def _mi_error(backend: str, fiber: StubFiber) -> float:
    n, dt = _N_MI, 1 / (_N_MI * _DF_MI)
    t = np.arange(n) * dt
    x0, y0 = 1e-7 * (0.3 + 0.2j), 1e-7 * (-0.1 + 0.25j)
    u = x0 * np.exp(2j * np.pi * _BIN_MI * _DF_MI * t) + np.conj(y0) * np.exp(
        -2j * np.pi * _BIN_MI * _DF_MI * t
    )
    out = get_propagator(backend).propagate(np.sqrt(_P_MI) + u, fiber, dt_s=dt)
    rotation = np.exp(-1j * fiber.gamma_per_w_per_km * 1e-3 * _P_MI * fiber.length_km * 1e3)
    spectrum = np.fft.fft(out * rotation) / n
    x, y = _linearised_sidebands(fiber, x0, y0)
    return max(
        abs(spectrum[_BIN_MI] - x) / abs(x), abs(np.conj(spectrum[n - _BIN_MI]) - y) / abs(y)
    )


@pytest.mark.parametrize("raman", RAMAN_MODELS)
@pytest.mark.parametrize(
    ("method", "shock", "steps", "order", "tol"),
    [
        ("ssfm", False, (100, 200, 400), 2, 2e-4),
        ("ssfm4-suzuki", False, (50, 100), 4, 3e-7),
        ("rk4ip", False, (100, 200, 400), 4, 1e-6),
        ("rk4ip", True, (100, 200, 400), 4, 1e-6),
    ],
)
def test_linearised_gnlse_matches_the_closed_form(
    nonlinear_backend, raman, method, shock, steps, order, tol
):
    """Stokes/anti-Stokes seeds converge to the exact solution at the scheme's order.

    (The floor, ≈1e-8, is the seeds' own nonlinearity, which the
    linearisation drops.)
    """
    errors = [_mi_error(nonlinear_backend, _mi_fiber(shock, method, k, raman)) for k in steps]
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine == pytest.approx(2**order, rel=0.15)
    assert errors[-1] < tol


def test_linearised_gnlse_has_raman_gain_on_the_stokes_side():
    """Positive baseband frequency is the lower optical frequency: it grows."""
    fiber = replace(_mi_fiber(False, "rk4ip", 1, RamanResponse()), beta2_s2_per_m=2e-27)
    x, y = _linearised_sidebands(fiber, 1e-7, 1e-7)
    gain = RamanResponse().gain_per_w_per_m(10.0, _BIN_MI * _DF_MI) * _P_MI * 500.0
    assert gain > 1
    assert np.log(abs(x) ** 2 / 1e-14) == pytest.approx(gain, rel=0.02)  # Δβ ≫ γP: pure Raman
    assert abs(y) ** 2 < 1e-14  # the anti-Stokes seed is absorbed


# ---------------------------------------------------------------------------
# Self-steepening and photon number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_db_per_km", [0.0, 0.5])
@pytest.mark.parametrize(("fraction", "steps"), [(0.3, (100, 200)), (0.6, (200, 400))])
def test_self_steepening_matches_the_characteristics_solution(
    nonlinear_backend, alpha_db_per_km, fraction, steps
):
    r"""Without dispersion :math:`\partial_zP + \alpha P + 3\gamma\tau P\,\partial_tP = 0`.

    So :math:`P(z, \xi + 3\gamma\tau P_0(\xi)L_\mathrm{eff}) = e^{-\alpha z}P_0(\xi)`
    until the shock at :math:`L_\mathrm{eff} = 1/(3\gamma\tau\max|P_0'|)`; the
    trailing edge steepens. Checked at ``fraction`` of that distance.
    """
    n, dt = 2**13, 1e-15
    t = (np.arange(n) - n // 2) * dt
    t0, tau, gamma, p0 = 200e-15, 10e-15, 10.0, 100.0
    g = gamma * 1e-3

    def power(s: np.ndarray) -> np.ndarray:
        return p0 * np.exp(-(s**2) / t0**2)

    slope = p0 * math.sqrt(2) / t0 * math.exp(-0.5)  # max |P₀'|
    leff = fraction / (3 * g * tau * slope)
    alpha = alpha_db_per_km * math.log(10) / 10 / 1e3
    z = -math.log1p(-alpha * leff) / alpha if alpha else leff
    xi = np.linspace(-20 * t0, 20 * t0, 2_000_001)
    exact = math.exp(-alpha * z) * np.interp(t, xi + 3 * g * tau * power(xi) * leff, power(xi))
    errors = []
    for k in steps:
        fiber = StubFiber(
            length_km=z / 1e3,
            alpha_db_per_km=alpha_db_per_km,
            beta2_s2_per_m=0.0,
            gamma_per_w_per_km=gamma,
            n_steps=k,
            scheme=StubScheme(method="rk4ip"),
            effective_shock_time_s=tau,
        )
        out = get_propagator(nonlinear_backend).propagate(
            np.sqrt(power(t)).astype(complex), fiber, dt_s=dt
        )
        errors.append(float(np.max(np.abs(np.abs(out) ** 2 - exact)) / p0))
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine == pytest.approx(16, rel=0.15)
    assert errors[-1] < 1e-8
    peak = np.argmax(np.abs(out))
    assert t[peak] > 0.1 * t0  # the peak is delayed


@pytest.mark.parametrize("raman", RAMAN_MODELS)
def test_photon_number_is_conserved_with_raman_and_self_steepening(nonlinear_backend, raman):
    r"""N = Σ|Ã(ω)|²/(ω₀ − ω) is invariant for τ = 1/ω₀; the energy is not.

    A third-order soliton (β₂, β₃) fissions and its fragments shift to the
    red, losing energy; RK4IP conserves N to its order.
    """
    n, dt = 2**12, 2e-15
    t = (np.arange(n) - n // 2) * dt
    omega = 2 * np.pi * np.fft.fftfreq(n, dt)
    t0, beta2, gamma = 50e-15, -2.17e-26, 1.27
    p0 = 9 * abs(beta2) / (gamma * 1e-3 * t0**2)
    a0 = np.fft.ifftshift(np.sqrt(p0) / np.cosh(t / t0)).astype(complex)
    omega0 = 2 * np.pi * C / 1550e-9

    def photons(a: np.ndarray) -> float:
        return float(np.sum(np.abs(np.fft.fft(a)) ** 2 / (omega0 - omega)))

    base = StubFiber(
        length_km=1.5 * t0**2 / abs(beta2) / 1e3,
        beta2_s2_per_m=beta2,
        beta3_s3_per_m=1.2e-40,
        gamma_per_w_per_km=gamma,
        scheme=StubScheme(method="rk4ip"),
        raman=raman,
        effective_shock_time_s=1 / omega0,
    )
    drift = []
    for k in (1000, 2000):
        out = get_propagator(nonlinear_backend).propagate(a0, replace(base, n_steps=k), dt_s=dt)
        drift.append(abs(photons(out) / photons(a0) - 1))
    assert drift[0] / drift[1] > 14 and drift[1] < 2e-7
    energy_change = np.sum(np.abs(out) ** 2) / np.sum(np.abs(a0) ** 2) - 1
    assert -0.1 < energy_change < -0.03  # the red-shifted light carries less energy


# ---------------------------------------------------------------------------
# Invertibility and cross-backend agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ssfm", "ssfm4-yoshida", "ssfm4-suzuki"])
def test_split_step_with_raman_is_exactly_invertible(nonlinear_backend, method):
    """The Raman-modified Kerr step is still a pure phase, so backward undoes forward."""
    fiber = StubFiber(
        length_km=30.0,
        alpha_db_per_km=0.2,
        beta3_s3_per_m=1.2e-40,
        higher_order_betas=(3e-55,),
        n_steps=20,
        scheme=StubScheme(method=method),
        raman=RamanResponse(model="lin-agrawal"),
    )
    x = _random_field((2048,), 0.1)
    prop = get_propagator(nonlinear_backend)
    back = prop.propagate(prop.propagate(x, fiber, 2e-13), fiber, 2e-13, direction="backward")
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-12 * np.max(np.abs(x)))


_GNLSE_CASES = {
    "ssfm-raman": {"raman": RamanResponse()},
    "log-steps-effective-raman": {
        "raman": RamanResponse(model="lin-agrawal"),
        "step_distribution": "logarithmic",
        "step_log_factor": 1 / 3,
        "scheme": StubScheme(nonlinear_position=0.3, kerr_length="effective"),
    },
    "filtered-raman": {
        "raman": RamanResponse(),
        "scheme": StubScheme(kerr_filter_bandwidth_hz=200e9),
    },
    "local-error-raman": {
        "raman": RamanResponse(),
        "scheme": StubScheme(method="ssfm-local-error", local_error=1e-6),
    },
    "rk4ip-raman-shock": {
        "raman": RamanResponse(model="lin-agrawal"),
        "scheme": StubScheme(method="rk4ip"),
        "effective_shock_time_s": 2e-15,
    },
    "loss-spectrum-and-betas": {
        "loss_spectrum": LossSpectrum(
            wavelengths_m=[1.54e-6, 1.56e-6], alpha_db_per_km=[0.25, 0.15]
        ),
        "higher_order_betas": (3e-55, -1e-68),
    },
}


@pytest.mark.parametrize("case", list(_GNLSE_CASES), ids=list(_GNLSE_CASES))
def test_gnlse_terms_agree_with_numpy_reference(nonlinear_backend, case):
    fiber = StubFiber(
        length_km=20.0,
        alpha_db_per_km=0.2,
        beta3_s3_per_m=1.2e-40,
        n_steps=30,
        **_GNLSE_CASES[case],
    )
    field = _random_field((2, 2048), 0.1)
    ref = get_propagator("numpy").propagate(field, fiber, dt_s=2e-13)
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=2e-13)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-11 * np.max(np.abs(ref)))
