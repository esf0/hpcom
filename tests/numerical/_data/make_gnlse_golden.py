"""Generate ``gnlse_python_v2_0.npz``: gnlse-python outputs for hpcom's GNLSE cross-check.

gnlse-python (MIT, https://github.com/WUST-FOG/gnlse-python) is an
independent Python port of Dudley, Genty & Coen's GNLSE code (Rev. Mod. Phys.
78, 1135, 2006): the interaction-picture equation in the frequency domain,
integrated by SciPy's adaptive ``solve_ivp``. Only its *outputs* are stored
(with the inputs that produced them); hpcom does not depend on or include
its code. Run it in a separate environment::

    uv venv gnl && VIRTUAL_ENV=$PWD/gnl uv pip install "setuptools<70" wheel numpy scipy
    VIRTUAL_ENV=$PWD/gnl uv pip install --no-build-isolation gnlse==2.0.0
    gnl/bin/python tests/numerical/_data/make_gnlse_golden.py

Case: the supercontinuum benchmark of Dudley et al. (2006), Fig. 3 — 835 nm,
50 fs FWHM sech pulse of 10 kW (soliton order ≈ 8.7) in 15 cm of photonic
crystal fibre with β₂…β₁₀, γ = 0.11 /W/m, Blow–Wood Raman response
(f_R = 0.18) and self-steepening (τ = 1/ω₀), no loss. Saved at z = L/3,
2L/3 and L (rows of ``analytic`` / ``sampled``; ``z_m`` also lists z = 0).

Two Raman responses are stored:

* ``analytic`` — gnlse-python run with the exact convolution kernel that
  hpcom uses (the inverse DFT of the closed-form :math:`H_R(\\omega)` on the
  grid), so any difference is due to the solvers alone;
* ``sampled`` — gnlse-python's own ``raman_blowwood`` (:math:`h_R(t)` sampled
  on its time grid), the conventional discretisation (stored as complex64:
  only its O(1) difference from ``analytic`` is used).
"""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path

import numpy as np
from gnlse import GNLSE, GNLSESetup
from gnlse.raman_response import raman_blowwood

OUT = Path(__file__).with_name("gnlse_python_v2_0.npz")
C_NM_PER_PS = 299_792.458
N, WINDOW_PS, WAVELENGTH_NM = 2**13, 12.5, 835.0
FWHM_PS, PEAK_W, LENGTH_M, GAMMA = 0.050, 10_000.0, 0.15, 0.11
#: β₂ … β₁₀ in psⁿ/m (Dudley et al. 2006).
BETAS_PS = (
    -11.830e-3,
    8.1038e-5,
    -9.5205e-8,
    2.0737e-10,
    -5.3943e-13,
    1.3486e-15,
    -2.5495e-18,
    3.0524e-21,
    -1.7140e-24,
)
TAU1_PS, TAU2_PS, F_R = 0.0122, 0.032, 0.18
Z_SAVES = 4
RTOL, ATOL = 1e-11, 1e-12


class _Taylor:
    """Lossless Taylor dispersion; gnlse's own class uses np.math, gone in NumPy 2."""

    def D(self, V: np.ndarray) -> np.ndarray:  # noqa: N802, N803 - gnlse's interface
        b = sum(beta / math.factorial(i + 2) * V ** (i + 2) for i, beta in enumerate(BETAS_PS))
        return 1j * b


def _analytic_raman(t_ps: np.ndarray) -> tuple[float, np.ndarray]:
    """Kernel whose gnlse convolution equals hpcom's Re IFFT(FFT(P)·H_R(ω))."""
    dt = t_ps[1] - t_ps[0]
    omega = 2 * np.pi * np.fft.fftfreq(t_ps.size, d=dt)  # rad/ps
    a, b = 1 / TAU2_PS, 1 / TAU1_PS
    h = (a * a + b * b) / ((a + 1j * omega) ** 2 + b * b)
    kernel = np.real(np.fft.ifft(h)) / dt  # 1/ps, index 0 = zero lag
    return F_R, np.fft.ifftshift(kernel)


def _run(raman) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    setup = GNLSESetup()
    setup.resolution = N
    setup.time_window = WINDOW_PS
    setup.wavelength = WAVELENGTH_NM
    setup.fiber_length = LENGTH_M
    setup.nonlinearity = GAMMA
    setup.z_saves = Z_SAVES
    setup.raman_model = raman
    setup.self_steepening = True
    setup.dispersion_model = _Taylor()
    setup.rtol, setup.atol, setup.method = RTOL, ATOL, "DOP853"
    t = np.linspace(-WINDOW_PS / 2, WINDOW_PS / 2, N)
    t0 = FWHM_PS / (2 * math.log(1 + math.sqrt(2)))
    a0 = math.sqrt(PEAK_W) / np.cosh(t / t0)
    setup.pulse_model = a0.astype(np.complex128)
    solution = GNLSE(setup).run()
    return a0, solution.At, solution.Z, float(t[1] - t[0])


def main() -> None:
    a0, analytic, z, dt_ps = _run(_analytic_raman)
    _, sampled, _, _ = _run(raman_blowwood)
    np.savez_compressed(
        OUT,
        gnlse_version=importlib.metadata.version("gnlse"),
        a0=a0,
        dt_s=dt_ps * 1e-12,
        z_m=z,
        analytic=analytic[1:],
        sampled=sampled[1:].astype(np.complex64),
        wavelength_m=WAVELENGTH_NM * 1e-9,
        gamma_per_w_per_km=GAMMA * 1e3,
        betas_s_per_m=np.array([b * 1e-12 ** (i + 2) for i, b in enumerate(BETAS_PS)]),
        tau1_s=TAU1_PS * 1e-12,
        tau2_s=TAU2_PS * 1e-12,
        f_r=F_R,
        rtol=RTOL,
        atol=ATOL,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
