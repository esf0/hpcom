"""Tests for the propagator registry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends import (
    Propagator,
    get_propagator,
    list_propagators,
    register_propagator,
)


@dataclass(frozen=True)
class _StubFiber:
    length_km: float = 80.0
    alpha_db_per_km: float = 0.2
    beta2_s2_per_m: float = -2.16e-26
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 10
    polarization: str = "single"


class _IdentityPropagator(Propagator):
    name = "identity-test"

    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        return field


def test_builtin_numpy_linear_is_autoregistered():
    assert "numpy-linear" in list_propagators()


def test_register_and_retrieve_instance():
    register_propagator("identity-test", _IdentityPropagator())
    prop = get_propagator("identity-test")
    assert isinstance(prop, _IdentityPropagator)


def test_register_accepts_class_and_instantiates():
    register_propagator("identity-test-cls", _IdentityPropagator)
    assert isinstance(get_propagator("identity-test-cls"), _IdentityPropagator)


def test_unknown_backend_raises_with_helpful_message():
    register_propagator("identity-test", _IdentityPropagator())
    with pytest.raises(KeyError, match="Unknown propagator backend"):
        get_propagator("definitely-not-a-real-backend")


def test_register_rejects_non_propagator():
    with pytest.raises(TypeError):
        register_propagator("bad", object())  # type: ignore[arg-type]


def test_abstract_propagator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Propagator()  # type: ignore[abstract]


def test_propagate_called_via_registry_returns_input_for_identity():
    register_propagator("identity-test", _IdentityPropagator())
    prop = get_propagator("identity-test")
    field = np.array([1 + 0j, 0 + 0j, 1 + 0j, 0 + 0j])
    out = prop.propagate(field, _StubFiber(), dt_s=1e-11)
    np.testing.assert_array_equal(out, field)
