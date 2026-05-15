"""Propagator backends and registry.

A :class:`~hpcom.backends.base.Propagator` is the seam between the user-facing
:class:`hpcom.fiber.Fiber` API and a concrete numerical implementation
(NumPy, TensorFlow, JAX in v1.1+, …).

Backends register themselves under a string name and the
:class:`~hpcom.fiber.Fiber` resolves them via :func:`get_propagator` at
``propagate()`` time. This lets users swap engines without touching scenario
code.
"""

from __future__ import annotations

from hpcom.backends.base import Propagator

__all__ = [
    "Propagator",
    "get_propagator",
    "list_propagators",
    "register_propagator",
]

_REGISTRY: dict[str, Propagator] = {}


def register_propagator(name: str, propagator: Propagator | type[Propagator]) -> None:
    """Register a propagator under ``name``.

    Accepts either an instance or a class (instantiated with no args). Replaces
    any prior registration at the same name silently — useful in tests.
    """
    if isinstance(propagator, type):
        propagator = propagator()
    if not isinstance(propagator, Propagator):
        raise TypeError(
            f"propagator must be a Propagator instance or subclass, got {type(propagator).__name__}"
        )
    _REGISTRY[name] = propagator


def get_propagator(name: str) -> Propagator:
    """Look up a registered propagator by name. Raises :class:`KeyError` if absent."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown propagator backend {name!r}. Registered: {available}.")
    return _REGISTRY[name]


def list_propagators() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def _autoregister_builtin_backends() -> None:
    """Register backends bundled with hpcom. Failures are silent (e.g. TF missing)."""
    try:
        from hpcom.backends.numpy import LinearNumpyPropagator

        register_propagator("numpy-linear", LinearNumpyPropagator())
    except ImportError:
        pass

    try:
        from hpcom.backends.tensorflow import TensorFlowPropagator

        register_propagator("tensorflow", TensorFlowPropagator())
    except ImportError:
        # TensorFlow is an optional extra (`pip install hpcom[tf]`).
        pass


_autoregister_builtin_backends()
