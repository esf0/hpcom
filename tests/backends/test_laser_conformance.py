"""Gain terms of fiber lasers (:mod:`hpcom.laser`) on every backend, against exact solutions.

``docs/physics.rst``, *Fiber lasers*:

* unsaturated gain of finite bandwidth: every frequency amplified by
  :math:`e^{g_0G(\\omega)L/2}` — exact in every scheme;
* saturated gain on a CW signal: the closed form
  :math:`\\ln(P/P_0) + (P - P_0)/P_\\mathrm{sat} = g_0L`, second order in the
  step in every split-step scheme;
* nonlinear gain or two-photon absorption without dispersion:
  :math:`P = P_0/(1 - 2\\varepsilon P_0z)` with the phase
  :math:`-\\gamma\\ln(1 - 2\\varepsilon P_0z)/2\\varepsilon` — exact, and run
  backwards exactly by the split-step schemes;
* the chirped dissipative soliton of the cubic complex Ginzburg–Landau
  equation (Pereira & Stenflo 1977): Strang splitting converges at second
  order, RK4IP at fourth; with bounded (Lorentzian) gain the fourth-order
  compositions keep their order too;
* every backend equals the NumPy reference.

Fibers are duck-typed (:class:`StubGainFiber`), so these tests pin the
backend contract, not :class:`hpcom.laser.GainFiber`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.optimize import brentq

from hpcom.backends import get_propagator
from tests.backends.test_propagator_conformance import StubFiber, StubScheme, _random_field

DB = math.log(10.0) / 10.0


@dataclass(frozen=True)
class StubGainFiber(StubFiber):
    """:class:`StubFiber` with the gain fields of :class:`hpcom.laser.GainFiber`."""

    gain_db_per_m: float = 0.0
    gain_bandwidth_hz: float | None = None
    gain_shape: str = "lorentzian"
    saturation_energy_j: float | None = None
    nonlinear_gain_per_w_per_km: float = 0.0


LASER = StubGainFiber(
    length_km=0.005,
    beta2_s2_per_m=-2e-26,
    gamma_per_w_per_km=3.0,
    gain_db_per_m=1.0,
    n_steps=20,
)


# ---------------------------------------------------------------------------
# Linear gain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["lorentzian", "parabolic"])
@pytest.mark.parametrize("method", ["ssfm", "rk4ip", "ssfm-local-error"])
def test_unsaturated_gain_amplifies_every_frequency_by_its_gain(any_backend, shape, method):
    r"""γ = 0: :math:`\tilde A(L) = \tilde A(0)\,e^{(i\beta_2\omega^2/2 + g_0G(\omega)/2)L}`."""
    n, dt, fwhm = 512, 1e-13, 3e12
    fiber = replace(
        LASER,
        gamma_per_w_per_km=0.0,
        gain_bandwidth_hz=fwhm,
        gain_shape=shape,
        scheme=StubScheme(method=method, local_error=1e-6 if "local" in method else None),
    )
    x = _random_field((n,), 0.3)
    f = np.fft.fftfreq(n, d=dt)
    x2 = (2 * f / fwhm) ** 2
    g = 1 / (1 + x2) if shape == "lorentzian" else 1 - x2 / 2
    omega = 2 * np.pi * f
    length = fiber.length_km * 1e3
    exponent = 1j * fiber.beta2_s2_per_m / 2 * omega**2 + fiber.gain_db_per_m * DB * g / 2
    expected = np.fft.ifft(np.fft.fft(x) * np.exp(exponent * length))
    out = get_propagator(any_backend).propagate(x, fiber, dt)
    np.testing.assert_allclose(out, expected, atol=1e-12 * np.max(np.abs(expected)))


# ---------------------------------------------------------------------------
# Saturated gain
# ---------------------------------------------------------------------------


def _saturated_amplifier_error(backend: str, n_steps: int, polarization: str, method: str) -> float:
    r"""CW through saturated gain: :math:`\ln(P/P_0) + (P - P_0)/P_\mathrm{sat} = g_0L`.

    :math:`E_\mathrm{sat} = P_\mathrm{sat}T` for a window :math:`T`; with
    two polarisations :math:`P` is their sum.
    """
    n, dt = 64, 1e-12
    p0, p_sat, length = 1e-3, 5e-3, 10.0
    fiber = replace(
        LASER,
        length_km=length / 1e3,
        beta2_s2_per_m=0.0,
        gamma_per_w_per_km=0.0,
        gain_db_per_m=2.0,
        saturation_energy_j=p_sat * n * dt,
        n_steps=n_steps,
        polarization=polarization,
        scheme=StubScheme(method=method),
    )
    g0l = fiber.gain_db_per_m * DB * length
    exact = brentq(lambda p: math.log(p / p0) + (p - p0) / p_sat - g0l, p0, 1.0)
    shape = (2, n) if polarization == "dual" else (n,)
    x = np.full(shape, math.sqrt(p0 / shape[0] if polarization == "dual" else p0), complex)
    out = get_propagator(backend).propagate(x, fiber, dt)
    power = np.sum(np.mean(np.abs(out) ** 2, axis=-1))
    return abs(power - exact) / exact


@pytest.mark.parametrize("method", ["ssfm", "ssfm4-yoshida", "ssfm4-suzuki"])
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_saturated_gain_converges_at_second_order_to_the_closed_form(
    nonlinear_backend, polarization, method
):
    """Second order in every split-step scheme: the mid-step energy update is second order."""
    errors = [
        _saturated_amplifier_error(nonlinear_backend, k, polarization, method) for k in (10, 20, 40)
    ]
    for coarse, fine in itertools.pairwise(errors):
        assert 3.5 < coarse / fine < 4.5
    assert errors[-1] < 1.5e-3


# ---------------------------------------------------------------------------
# Nonlinear gain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [-5.0, 0.4], ids=["two-photon-absorption", "nonlinear-gain"])
@pytest.mark.parametrize(
    "scheme",
    [
        StubScheme(),
        StubScheme(method="ssfm4-suzuki"),
        StubScheme(method="ssfm", kerr_length="effective"),
        StubScheme(method="ssfm-local-error", local_error=1e-6),
    ],
    ids=["ssfm", "suzuki", "effective-length", "local-error"],
)
def test_nonlinear_gain_without_dispersion_is_exact(nonlinear_backend, eps, scheme):
    r"""∂zA = (iγ + ε)|A|²A: :math:`P = P_0/(1 - 2\varepsilon P_0z)`,
    :math:`\phi = -\gamma\ln(1 - 2\varepsilon P_0z)/2\varepsilon`; backwards too."""
    fiber = StubGainFiber(
        length_km=1.0,
        beta2_s2_per_m=0.0,
        gamma_per_w_per_km=1.3,
        nonlinear_gain_per_w_per_km=eps,
        n_steps=3,
        scheme=scheme,
    )
    x = np.sqrt(np.linspace(0.01, 0.3, 64)).astype(complex)
    p0, z = np.abs(x) ** 2, 1e3
    u = 2 * eps * 1e-3 * p0 * z
    expected = np.sqrt(p0 / (1 - u)) * np.exp(-1j * 1.3e-3 * np.log1p(-u) / (2 * eps * 1e-3))
    prop = get_propagator(nonlinear_backend)
    out = prop.propagate(x, fiber, 1e-12)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0)
    if scheme.method != "ssfm-local-error":
        back = prop.propagate(out, fiber, 1e-12, direction="backward")
        np.testing.assert_allclose(back, x, rtol=1e-12, atol=0)


def _cgle_soliton() -> tuple[StubGainFiber, float, np.ndarray, np.ndarray, float]:
    r"""Pereira–Stenflo chirped soliton of the cubic CGLE in the GainFiber parameters.

    :math:`A_z = \delta A + (b - i\beta)A_{tt} + (\varepsilon + i\gamma)|A|^2A`,
    :math:`\beta = \beta_2/2`, :math:`b = g_0/(4\Omega_g^2)` (parabolic gain),
    :math:`\delta = (g_0 - \alpha)/2`. With
    :math:`A = a\,\mathrm{sech}(t/T)^{1+id}e^{i\kappa z}`:
    :math:`A_2d^2 + A_1d - 2A_2 = 0` (:math:`A_2 = \varepsilon\beta + \gamma b`,
    :math:`A_1 = 3(\varepsilon b - \gamma\beta)`),
    :math:`a^2T^2 = [b(2 - d^2) + 3\beta d]/\varepsilon`,
    :math:`T^2 = -[b(1 - d^2) + 2\beta d]/\delta`,
    :math:`\kappa = [2bd - \beta(1 - d^2)]/T^2`. Returns the fiber (three
    dispersion lengths), dt, the input, the exact output and the peak amplitude.
    """
    beta2, fwhm, g0, gamma, eps = -2e-26, 1e12, 0.4, 1e-3, 0.3e-3
    beta, b = beta2 / 2, g0 / (4 * (math.pi * fwhm) ** 2)
    alpha = 1.2 * g0
    delta = (g0 - alpha) / 2
    a2, a1 = eps * beta + gamma * b, 3 * (eps * b - gamma * beta)
    d = (-a1 + math.sqrt(a1**2 + 8 * a2**2)) / (2 * a2)
    t2 = -(b * (1 - d * d) + 2 * beta * d) / delta
    amp = math.sqrt((b * (2 - d * d) + 3 * beta * d) / eps / t2)
    kappa = (2 * b * d - beta * (1 - d * d)) / t2
    period = math.sqrt(t2)
    n, dt = 2048, period / 40
    t = np.fft.ifftshift((np.arange(n) - n // 2) * dt)
    length = 3 * t2 / abs(beta2)
    pulse = amp * (1 / np.cosh(t / period)) ** (1 + 1j * d)
    fiber = StubGainFiber(
        length_km=length / 1e3,
        alpha_db_per_km=alpha / DB * 1e3,
        beta2_s2_per_m=beta2,
        gamma_per_w_per_km=gamma * 1e3,
        nonlinear_gain_per_w_per_km=eps * 1e3,
        gain_db_per_m=g0 / DB,
        gain_bandwidth_hz=fwhm,
        gain_shape="parabolic",
    )
    return fiber, dt, pulse, pulse * np.exp(1j * kappa * length), amp


@pytest.mark.parametrize(("method", "order"), [("ssfm", 2), ("rk4ip", 4)])
def test_complex_ginzburg_landau_soliton_converges_at_the_scheme_order(
    nonlinear_backend, method, order
):
    fiber, dt, pulse, exact, amp = _cgle_soliton()
    prop = get_propagator(nonlinear_backend)
    errors = [
        np.max(
            np.abs(
                prop.propagate(
                    pulse, replace(fiber, n_steps=k, scheme=StubScheme(method=method)), dt
                )
                - exact
            )
        )
        / amp
        for k in (50, 100, 200)
    ]
    for coarse, fine in itertools.pairwise(errors):
        assert 0.8 * 2**order < coarse / fine < 1.2 * 2**order
    assert errors[-1] < (5e-4 if order == 2 else 2e-7)


@pytest.mark.parametrize(
    ("method", "order"),
    [("ssfm", 2), ("ssfm4-yoshida", 4), ("ssfm4-suzuki", 4), ("rk4ip", 4)],
)
def test_bounded_lorentzian_gain_keeps_every_scheme_at_its_order(nonlinear_backend, method, order):
    """A soliton amplified 3× through 2 THz of Lorentzian gain (≈1 rad of SPM).

    Self-convergence: :math:`\\|A_K - A_{2K}\\| / \\|A_{2K} - A_{4K}\\| \\to 2^p`.
    (Parabolic gain is unbounded; GainFiber rejects the compositions there.)
    """
    n, dt, t0 = 2048, 10e-15, 0.3e-12
    t = np.fft.ifftshift((np.arange(n) - n // 2) * dt)
    pulse = np.sqrt(2e-26 / (3e-3 * t0**2)) / np.cosh(t / t0) + 0j  # N = 1
    fiber = replace(LASER, gain_bandwidth_hz=2e12, scheme=StubScheme(method=method))
    prop = get_propagator(nonlinear_backend)
    runs = [prop.propagate(pulse, replace(fiber, n_steps=k), dt) for k in (50, 100, 200)]
    ratio = np.max(np.abs(runs[0] - runs[1])) / np.max(np.abs(runs[1] - runs[2]))
    assert 0.85 * 2**order < ratio < 1.15 * 2**order


# ---------------------------------------------------------------------------
# Backends agree
# ---------------------------------------------------------------------------

LASER_CASES = {
    "nonlinear-gain-lorentzian": replace(
        LASER, gain_bandwidth_hz=2e12, nonlinear_gain_per_w_per_km=0.5
    ),
    "cgle-rk4ip": replace(
        LASER,
        gain_bandwidth_hz=2e12,
        gain_shape="parabolic",
        nonlinear_gain_per_w_per_km=0.5,
        scheme=StubScheme(method="rk4ip"),
    ),
    "two-photon-absorption-local-error": replace(
        LASER,
        gain_db_per_m=0.0,
        nonlinear_gain_per_w_per_km=-1.0,
        scheme=StubScheme(method="ssfm-local-error", local_error=1e-6),
    ),
    "saturated-dual-log": replace(
        LASER,
        polarization="dual",
        gain_bandwidth_hz=2e12,
        saturation_energy_j=1e-12,
        step_distribution="logarithmic",
    ),
    "saturated-yoshida-nonlinear-gain": replace(
        LASER,
        saturation_energy_j=1e-12,
        nonlinear_gain_per_w_per_km=0.3,
        scheme=StubScheme(method="ssfm4-yoshida"),
    ),
}


@pytest.mark.parametrize("name", list(LASER_CASES))
def test_every_backend_equals_numpy(nonlinear_backend, name):
    fiber = LASER_CASES[name]
    shape = (3, 2, 256) if fiber.polarization == "dual" else (3, 256)  # batched
    x = _random_field(shape, 0.5)
    ref = get_propagator("numpy").propagate(x, fiber, 1e-13)
    out = get_propagator(nonlinear_backend).propagate(x, fiber, 1e-13)
    np.testing.assert_allclose(out, ref, atol=1e-12 * np.max(np.abs(ref)))
    if fiber.saturation_energy_j is not None:  # saturation acts per batch element
        one = get_propagator(nonlinear_backend).propagate(x[1], fiber, 1e-13)
        np.testing.assert_allclose(out[1], one, atol=1e-12 * np.max(np.abs(one)))


def test_the_linear_backend_rejects_saturated_gain():
    fiber = replace(LASER, saturation_energy_j=1e-12)
    with pytest.raises(ValueError, match="saturated gain"):
        get_propagator("numpy-linear").propagate(np.ones(64, complex), fiber, 1e-12)
