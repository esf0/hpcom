"""Cross-checks against OptiCommPy, an independent implementation of the same models.

Reference outputs of OptiCommPy 0.10.0 are stored in
``_data/opticommpy_v0_10.npz`` together with their inputs (generator:
``_data/make_opticommpy_golden.py``; OptiCommPy needs numba, so it runs in a
separate environment). With ``optic`` importable, :func:`test_live_opticommpy`
also compares against a fresh run.

* Scalar NLSE — OptiCommPy's ``ssfm`` is the same symmetric split-step
  scheme, so the fields agree to rounding at every step size.
* Manakov — ``manakovSSF`` uses an iterated trapezoidal nonlinear step, a
  different scheme: both converge (second order) to the same solution.
* EDFA — ``edfa`` implements the same Essiambre ASE model: the per-sample
  noise variance agrees within the statistical uncertainty.
"""

from __future__ import annotations

import functools
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy import constants

from hpcom import EDFA, Fiber, Link

DATA = np.load(Path(__file__).parent / "_data" / "opticommpy_v0_10.npz")
FS, FC = float(DATA["fs_hz"]), float(DATA["fc_hz"])
WAVELENGTH_M = constants.c / FC


def _link(polarization: str, n_steps: int, backend: str = "numpy") -> Link:
    """2 × 80 km, D = 16 ps/(nm·km), 0.2 dB/km, γ = 1.3 /(W·km), ideal amplifiers."""
    fiber = Fiber.ssmf(
        length_km=80.0,
        n_steps=n_steps,
        polarization=polarization,
        backend=backend,
        dispersion_ps_per_nm_per_km=16.0,
        wavelength_m=WAVELENGTH_M,
    ).replace(alpha_db_per_km=0.2, gamma_per_w_per_km=1.3)
    return Link.from_spans(2, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("hz_km", [8.0, 0.5])
def test_scalar_ssfm_equals_opticommpy_to_rounding(nonlinear_backend, hz_km):
    link = _link("single", round(80 / hz_km), nonlinear_backend)
    out = link.propagate(DATA["ssfm_input"], 1 / FS)
    assert _rel(out, DATA[f"ssfm_h{hz_km}"]) < 1e-12


@functools.cache
def _converged_manakov(backend: str) -> np.ndarray:
    return np.asarray(_link("dual", 2560, backend).propagate(DATA["manakov_input"].T, 1 / FS))


def test_manakov_and_opticommpy_converge_to_the_same_solution(nonlinear_backend):
    reference = _converged_manakov(nonlinear_backend)  # its own step error is ≈ 3e-7
    ocp = [_rel(DATA[f"manakov_h{hz}"].T, reference) for hz in (2.0, 0.5, 0.125)]
    for coarse, fine in itertools.pairwise(ocp):  # 4.2e-3 → 2.4e-4 → 1.5e-5
        assert 12 < coarse / fine < 20  # second order in OptiCommPy's step
    assert ocp[-1] < 3e-5
    # hpcom at a moderate step count is as close to OptiCommPy's finest as OptiCommPy itself.
    ours = _link("dual", 640, nonlinear_backend).propagate(DATA["manakov_input"].T, 1 / FS)
    assert _rel(ours, DATA["manakov_h0.125"].T) < 3e-5


def test_edfa_ase_variance_matches_opticommpy():
    n = int(DATA["n_edfa"])
    for gain_db, nf_db, measured in zip(
        DATA["edfa_gain_db"], DATA["edfa_nf_db"], DATA["edfa_var"], strict=True
    ):
        edfa = EDFA(gain_db=gain_db, noise_figure_db=nf_db, wavelength_m=WAVELENGTH_M)
        ratio = edfa.ase_variance_per_sample(1 / FS) / measured
        assert abs(ratio - 1) < 4 / np.sqrt(n), (gain_db, nf_db, ratio)  # |n|² has σ/μ = 1


def test_live_opticommpy():
    """Fresh OptiCommPy run (skipped unless ``optic`` is importable)."""
    channels = pytest.importorskip("optic.models.channels")
    utils = pytest.importorskip("optic.utils")
    param = utils.parameters()
    param.Ltotal, param.Lspan, param.hz = 160.0, 80.0, 4.0
    param.alpha, param.D, param.gamma, param.Fc, param.Fs = 0.2, 16.0, 1.3, FC, FS
    param.amp, param.prgsBar = "ideal", False
    x = DATA["ssfm_input"]
    expected = channels.ssfm(x.copy(), param)
    assert _rel(_link("single", 20).propagate(x, 1 / FS), expected) < 1e-12
