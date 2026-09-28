"""Fiber span configuration and propagation entry point."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import Fiber
from hpcom.units import dispersion_slope_to_beta3, dispersion_to_beta2


def test_construct_with_explicit_parameters():
    f = Fiber(
        length_km=50.0,
        alpha_db_per_km=0.25,
        beta2_s2_per_m=-2e-26,
        beta3_s3_per_m=1e-40,
        gamma_per_w_per_km=1.5,
        n_steps=10,
    )
    assert (f.length_km, f.alpha_db_per_km, f.beta2_s2_per_m, f.beta3_s3_per_m) == (
        50.0,
        0.25,
        -2e-26,
        1e-40,
    )
    assert f.backend == "numpy" and f.polarization == "single"


def test_ssmf_preset():
    f = Fiber.ssmf(80.0)
    assert f.alpha_db_per_km == 0.2
    assert f.gamma_per_w_per_km == 1.27
    assert f.beta2_s2_per_m == pytest.approx(dispersion_to_beta2(17.0))
    assert f.beta3_s3_per_m == 0.0


def test_ssmf_preset_options():
    f = Fiber.ssmf(
        80.0,
        polarization="dual",
        backend="numpy-linear",
        n_steps=7,
        dispersion_slope_ps_per_nm2_per_km=0.057,
    )
    assert (f.polarization, f.backend, f.n_steps) == ("dual", "numpy-linear", 7)
    assert f.beta3_s3_per_m == pytest.approx(dispersion_slope_to_beta3(0.057, 17.0))


def test_derived_quantities():
    f = Fiber.ssmf(80.0)
    assert f.loss_db == pytest.approx(16.0)
    alpha = 0.2 * np.log(10) / 10
    assert f.effective_length_km == pytest.approx((1 - np.exp(-alpha * 80)) / alpha)
    assert f.replace(alpha_db_per_km=0.0).effective_length_km == 80.0
    assert f.accumulated_dispersion_s2 == pytest.approx(f.beta2_s2_per_m * 80e3)


def test_with_backend_returns_a_new_instance():
    f = Fiber.ssmf(80.0)
    g = f.with_backend("tensorflow")
    assert g.backend == "tensorflow" and f.backend == "numpy"
    assert g.replace(backend="numpy") == f


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"length_km": 0.0}, "length_km"),
        ({"length_km": -1.0}, "length_km"),
        ({"length_km": 1.0, "alpha_db_per_km": -0.1}, "alpha_db_per_km"),
        ({"length_km": 1.0, "n_steps": 0}, "n_steps"),
        ({"length_km": 1.0, "gamma_per_w_per_km": -1.0}, "gamma_per_w_per_km"),
        ({"length_km": 1.0, "polarization": "both"}, "polarization"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, field):
    with pytest.raises(ValidationError, match=field):
        Fiber(**kwargs)


def test_propagate_and_apply_agree_and_preserve_shape():
    f = Fiber.ssmf(10.0, n_steps=5)
    x = np.exp(1j * np.linspace(0, 6, 256)) * 1e-2
    out = f.propagate(x, dt_s=1e-11)
    assert out.shape == x.shape and out.dtype == np.complex128
    np.testing.assert_array_equal(f.apply(x, 1e-11, rng=np.random.default_rng(0)), out)


def test_apply_inverse_undoes_apply():
    f = Fiber.ssmf(80.0, n_steps=20, polarization="dual")
    x = np.random.default_rng(0).standard_normal((2, 512)) * 3e-2 + 0j
    back = f.apply_inverse(f.apply(x, 1e-12), 1e-12)
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-13)


def test_unknown_backend_fails_at_propagation_time():
    f = Fiber(length_km=1.0, backend="no-such-backend")
    with pytest.raises(KeyError, match="no-such-backend"):
        f.propagate(np.ones(8, complex), 1e-12)
