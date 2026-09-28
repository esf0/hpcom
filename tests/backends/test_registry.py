"""Propagator registry: registration forms, lazy loading, error reporting."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

import hpcom.backends as backends
from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.backends import (
    BackendUnavailableError,
    Propagator,
    available_propagators,
    get_propagator,
    list_propagators,
    register_propagator,
)


@dataclass(frozen=True)
class _StubFiber:
    length_km: float = 80.0
    alpha_db_per_km: float = 0.2
    beta2_s2_per_m: float = -2.16e-26
    beta3_s3_per_m: float = 0.0
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 10
    polarization: str = "single"


class _IdentityPropagator(Propagator):
    name = "identity-test"

    def propagate(
        self, field: ComplexArray, fiber: FiberLike, dt_s: float, direction: Direction = "forward"
    ) -> ComplexArray:
        return field


@pytest.fixture(autouse=True)
def _restore_registry():
    factories, instances = dict(backends._factories), dict(backends._instances)
    yield
    backends._factories.clear()
    backends._factories.update(factories)
    backends._instances.clear()
    backends._instances.update(instances)


@pytest.mark.parametrize("name", ["numpy", "numpy-linear", "tensorflow"])
def test_builtin_backends_are_registered(name):
    assert name in list_propagators()


@pytest.mark.parametrize("name", ["numpy", "numpy-linear"])
def test_numpy_backends_are_always_available(name):
    assert name in available_propagators()


def test_register_and_retrieve_instance():
    prop = _IdentityPropagator()
    register_propagator("identity-test", prop)
    assert get_propagator("identity-test") is prop


def test_register_class_is_instantiated_once_lazily():
    register_propagator("identity-cls", _IdentityPropagator)
    first = get_propagator("identity-cls")
    assert isinstance(first, _IdentityPropagator)
    assert get_propagator("identity-cls") is first


def test_register_factory():
    calls = []

    def factory() -> Propagator:
        calls.append(1)
        return _IdentityPropagator()

    register_propagator("identity-factory", factory)
    assert calls == []  # lazy
    get_propagator("identity-factory")
    get_propagator("identity-factory")
    assert calls == [1]


def test_reregistering_replaces_cached_instance():
    register_propagator("x", _IdentityPropagator())
    old = get_propagator("x")
    register_propagator("x", _IdentityPropagator)
    assert get_propagator("x") is not old


def test_unknown_backend_raises_key_error_listing_registered():
    with pytest.raises(KeyError, match=r"Unknown propagator backend 'nope'.*numpy"):
        get_propagator("nope")


@pytest.mark.parametrize("bad", [object(), 42, "numpy"])
def test_register_rejects_non_propagators(bad):
    with pytest.raises(TypeError):
        register_propagator("bad", bad)  # type: ignore[arg-type]


def test_register_rejects_non_propagator_class():
    with pytest.raises(TypeError, match="not a Propagator subclass"):
        register_propagator("bad", dict)  # type: ignore[arg-type]


def test_factory_returning_wrong_type_is_rejected_on_use():
    register_propagator("bad-factory", lambda: object())  # type: ignore[arg-type, return-value]
    with pytest.raises(TypeError, match="returned object"):
        get_propagator("bad-factory")


def test_missing_optional_dependency_gives_install_hint():
    backends._factories["ghost"] = backends._builtin_factory(
        "hpcom_no_such_module", "Ghost", 'install it with `pip install "hpcom[ghost]"`', "ghost"
    )
    with pytest.raises(BackendUnavailableError, match=r"'ghost' is unavailable.*hpcom\[ghost\]"):
        get_propagator("ghost")
    assert "ghost" not in available_propagators()


def test_abstract_propagator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Propagator()  # type: ignore[abstract]


def test_registered_propagator_is_called_through_registry():
    register_propagator("identity-test", _IdentityPropagator())
    field = np.array([1 + 0j, 0, 1, 0])
    np.testing.assert_array_equal(
        get_propagator("identity-test").propagate(field, _StubFiber(), 1e-11), field
    )


def test_repr_names_backend():
    assert "numpy" in repr(get_propagator("numpy"))


@pytest.mark.parametrize(
    "statement",
    [
        "import hpcom",
        "from hpcom import Fiber, Simulation, SingleCarrierTransmitter",
        "import hpcom; hpcom.Fiber(length_km=1).propagate(__import__('numpy').ones(8, complex), 1e-12)",
    ],
)
def test_importing_hpcom_does_not_import_tensorflow_or_legacy(statement):
    """TensorFlow, scikit-learn and the legacy package load only on demand, and nothing warns."""
    code = (
        "import sys, warnings\n"
        "warnings.simplefilter('error')\n"
        f"{statement}\n"
        "bad = [m for m in ('tensorflow', 'sklearn', 'matplotlib', 'hpcom._legacy') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
