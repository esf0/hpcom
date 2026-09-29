"""Generalised NLSE: soliton self-frequency shift, recentring, gnlse-python cross-check.

Backend-level checks against exact solutions are in
``tests/backends/test_gnlse_conformance.py``; these use :class:`hpcom.Fiber`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hpcom import Fiber, LossSpectrum, RamanResponse
from hpcom.backends import available_propagators

N_SOL, DT_SOL, T0_SOL = 4096, 2e-15, 50e-15
BETA2, GAMMA = -2.17e-26, 1.27


def _omega(n: int, dt: float) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(n, dt)


def _centroid(a: np.ndarray, dt: float) -> float:
    """Mean angular frequency ⟨ω⟩ of the spectrum (hpcom's baseband convention)."""
    s = np.abs(np.fft.fft(a)) ** 2
    return float(np.sum(_omega(a.size, dt) * s) / np.sum(s))


def _raman_rate(a: np.ndarray, dt: float, raman: RamanResponse) -> float:
    r"""d⟨ω⟩/dz = γ f_R ∫P ∂_t(h_R*P) dt / ∫P dt — exact for the GNLSE without shock or loss.

    (Dispersion does not change |Ã(ω)|²; the instantaneous Kerr term gives
    ∫P∂_tP = 0.)
    """
    w = _omega(a.size, dt)
    p = np.abs(a) ** 2
    q = raman.f_r * np.real(np.fft.ifft(np.fft.fft(p) * raman.transfer(w)))
    dq = np.real(np.fft.ifft(1j * w * np.fft.fft(q)))
    return float(GAMMA * 1e-3 * np.sum(p * dq) / np.sum(p))


def _soliton() -> np.ndarray:
    t = (np.arange(N_SOL) - N_SOL // 2) * DT_SOL
    p0 = abs(BETA2) / (GAMMA * 1e-3 * T0_SOL**2)
    return (np.sqrt(p0) / np.cosh(t / T0_SOL)).astype(complex)


def _soliton_fiber(length_m: float, raman: RamanResponse, n_steps: int) -> Fiber:
    return Fiber(
        length_km=length_m / 1e3,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=BETA2,
        gamma_per_w_per_km=GAMMA,
        raman=raman,
        n_steps=n_steps,
        scheme="rk4ip",
    )


@pytest.mark.parametrize("model", ["blow-wood", "lin-agrawal"])
def test_mean_frequency_moves_at_the_exact_raman_rate(model):
    """The identity d⟨ω⟩/dz = γ∫P∂_tQ/E, at z = 3 L_D, by a centred difference."""
    raman = RamanResponse(model=model)
    l_d = T0_SOL**2 / abs(BETA2)
    a0 = _soliton()
    z, dz = 3 * l_d, 0.02 * l_d
    mid = _soliton_fiber(z, raman, 600).propagate(a0, DT_SOL)
    step = _soliton_fiber(dz, raman, 8)
    ahead = step.propagate(mid, DT_SOL)
    behind = step.propagate(mid, DT_SOL, direction="backward")
    rate = (_centroid(ahead, DT_SOL) - _centroid(behind, DT_SOL)) / (2 * dz)
    assert rate == pytest.approx(_raman_rate(mid, DT_SOL, raman), rel=1e-5)


def test_soliton_self_frequency_shift_follows_gordon():
    """A fundamental soliton red-shifts linearly in z (Gordon 1986; Mitschke & Mollenauer 1986).

    The adiabatic rate is the exact Raman rate of a sech² pulse, constant as
    long as the soliton keeps its shape; for long pulses it tends to Gordon's
    :math:`8T_R|\\beta_2|/(15T_0^4)` with :math:`T_R = f_R\\int t\\,h_R\\,dt`.
    """
    raman = RamanResponse()
    l_d = T0_SOL**2 / abs(BETA2)
    a0 = _soliton()
    rate0 = _raman_rate(a0, DT_SOL, raman)
    assert rate0 > 0  # positive baseband frequency = longer wavelength: a red shift
    for z_over_ld in (2, 4, 8):
        z = z_over_ld * l_d
        out = _soliton_fiber(z, raman, 50 * z_over_ld).propagate(a0, DT_SOL)
        assert _centroid(out, DT_SOL) == pytest.approx(rate0 * z, rel=0.015)
        assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(a0) ** 2), rel=1e-8)
    # Gordon's linear-gain-slope formula: exact in the long-pulse limit.
    for t0, rel in ((50e-15, 0.3), (500e-15, 0.02), (2e-12, 0.002)):
        n, dt = 2**14, t0 / 20
        t = (np.arange(n) - n // 2) * dt
        pulse = (1 / np.cosh(t / t0)).astype(complex) * math.sqrt(
            abs(BETA2) / (GAMMA * 1e-3 * t0**2)
        )
        gordon = 8 * raman.response_time_s * abs(BETA2) / (15 * t0**4)
        assert _raman_rate(pulse, dt, raman) == pytest.approx(gordon, rel=rel)


def test_recentred_fiber_propagates_a_shifted_field_exactly():
    """Field·e^{iω_k t} through the fiber ≡ e^{iω_k t}·e^{iD(ω_k)L}·(field through the recentred fiber).

    Every term: β₁…β₆, loss spectrum, Raman, self-steepening (RK4IP). Exact as
    long as the field and its Kerr products stay inside the sampled band on
    both grids.
    """
    n, dt = 2**12, 5e-15
    t = np.arange(n) * dt
    rng = np.random.default_rng(3)
    spectrum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    spectrum[np.abs(np.fft.fftfreq(n)) > 0.02] = 0  # Kerr products stay inside both grids
    x = np.fft.ifft(spectrum)
    x *= np.sqrt(5.0 / np.mean(np.abs(x) ** 2))
    fiber = Fiber(
        length_km=0.05,
        beta1_s_per_m=1e-13,
        beta2_s2_per_m=-2e-26,
        beta3_s3_per_m=1e-40,
        higher_order_betas=(-2e-55, 3e-70, 1e-84),
        gamma_per_w_per_km=5.0,
        raman=RamanResponse(model="lin-agrawal"),
        self_steepening=True,
        scheme="rk4ip",
        n_steps=40,
    ).with_loss_spectrum(LossSpectrum(wavelengths_m=[1.4e-6, 1.7e-6], alpha_db_per_km=[40.0, 10.0]))
    f_k = 200 / (n * dt)  # a grid frequency: 9.8 THz
    carrier = np.exp(2j * np.pi * f_k * t)
    direct = fiber.propagate(x * carrier, dt)
    phase = np.exp(1j * fiber.dispersion_phase_per_m(f_k) * fiber.length_m)
    moved = fiber.recentred(f_k).propagate(x, dt) * carrier * phase
    np.testing.assert_allclose(moved, direct, rtol=0, atol=1e-11 * np.max(np.abs(direct)))


GNLSE_GOLDEN = Path(__file__).parent / "_data" / "gnlse_python_v2_0.npz"


@pytest.mark.slow
def test_matches_gnlse_python_on_the_dudley_supercontinuum():
    """Dudley et al. (2006) Fig. 3 benchmark vs gnlse-python 2.0 (``_data/make_gnlse_golden.py``).

    835 nm, 50 fs sech of 10 kW (soliton order ≈ 8.7), 15 cm of PCF with
    β₂…β₁₀, Blow–Wood Raman, self-steepening. gnlse-python integrates the
    same equation in the frequency domain with an adaptive RK method
    (DOP853, rtol 1e-11), here with hpcom's exact Raman kernel; RK4IP with
    16 000 steps agrees with it to 7e-5, 2e-4 and 5e-4 (relative L2 of the
    complex field) at L/3, 2L/3 and L — the level set by the reference, not
    by the step. gnlse-python's own sampled Raman response (dt = 1.5 fs)
    changes the field by O(1): a supercontinuum is that sensitive.
    """
    with np.load(GNLSE_GOLDEN) as data:
        d = {k: data[k] for k in data.files}
    betas = d["betas_s_per_m"]
    z = d["z_m"]
    backend = "jax" if "jax" in available_propagators() else "numpy"
    fiber = Fiber(
        length_km=float(z[1] - z[0]) / 1e3,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=betas[0],
        beta3_s3_per_m=betas[1],
        higher_order_betas=tuple(betas[2:]),
        gamma_per_w_per_km=float(d["gamma_per_w_per_km"]),
        wavelength_m=float(d["wavelength_m"]),
        raman=RamanResponse(
            tau1_s=float(d["tau1_s"]), tau2_s=float(d["tau2_s"]), fraction=float(d["f_r"])
        ),
        self_steepening=True,
        scheme="rk4ip",
        n_steps=16_000 // (z.size - 1),
        backend=backend,
    )
    a = d["a0"].astype(complex)
    dt = float(d["dt_s"])
    for i, tol in zip(range(1, z.size), (2e-4, 4e-4, 1.2e-3), strict=True):
        a = fiber.propagate(a, dt)
        ref = d["analytic"][i - 1]
        assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < tol
        spectrum, ref_spectrum = np.abs(np.fft.fft(a)) ** 2, np.abs(np.fft.fft(ref)) ** 2
        assert np.linalg.norm(spectrum - ref_spectrum) / np.linalg.norm(ref_spectrum) < tol
        sampled = d["sampled"][i - 1]
        assert np.linalg.norm(sampled - ref) / np.linalg.norm(ref) > 0.3
