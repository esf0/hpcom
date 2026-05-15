"""Tests for :class:`hpcom.fiber.Fiber`."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.fiber import Fiber


def test_construct_with_explicit_params():
    fiber = Fiber(length_km=80.0, backend="numpy-linear")
    assert fiber.length_km == 80.0
    assert fiber.backend == "numpy-linear"
    assert fiber.n_steps == 100


def test_ssmf_preset_has_expected_defaults():
    fiber = Fiber.ssmf(length_km=100.0, backend="numpy-linear")
    assert fiber.alpha_db_per_km == pytest.approx(0.2)
    assert fiber.gamma_per_w_per_km == pytest.approx(1.27)
    # D = 17 ps/nm/km at 1550 nm gives β₂ ≈ -2.17e-26 s²/m
    assert fiber.beta2_s2_per_m == pytest.approx(-2.17e-26, rel=5e-3)


def test_with_backend_returns_new_instance():
    fiber = Fiber.ssmf(80.0, backend="numpy-linear")
    swapped = fiber.with_backend("tensorflow")
    assert swapped is not fiber
    assert swapped.backend == "tensorflow"
    assert fiber.backend == "numpy-linear"


def test_invalid_length_rejected():
    with pytest.raises(ValueError, match="length_km"):
        Fiber(length_km=0.0)
    with pytest.raises(ValueError, match="length_km"):
        Fiber(length_km=-10.0)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError, match="alpha"):
        Fiber(length_km=80.0, alpha_db_per_km=-0.1)


def test_invalid_n_steps_rejected():
    with pytest.raises(ValueError, match="n_steps"):
        Fiber(length_km=80.0, n_steps=0)


def test_propagate_preserves_shape_and_dtype_via_numpy_backend():
    fiber = Fiber.ssmf(50.0, backend="numpy-linear")
    field = np.ones(256, dtype=np.complex128)
    out = fiber.propagate(field, dt_s=1e-11)
    assert out.shape == field.shape
    assert out.dtype == field.dtype


def test_frozen_dataclass_rejects_mutation():
    fiber = Fiber.ssmf(80.0)
    # dataclasses raises FrozenInstanceError, which subclasses AttributeError.
    with pytest.raises(AttributeError):
        fiber.length_km = 100.0  # type: ignore[misc]
