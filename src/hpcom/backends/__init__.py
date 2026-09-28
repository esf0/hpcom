"""Numerical backends: fiber propagators (and, in :mod:`hpcom.backends.kernels`, DSP kernels).

A :class:`~hpcom.backends.base.Propagator` is the seam between the
user-facing :class:`hpcom.fiber.Fiber` and a concrete numerical engine.
Propagators are registered under a string name and resolved by
:func:`get_propagator` when a fiber propagates. Built-in backends are
registered *lazily* — as import paths — so ``import hpcom`` never imports
TensorFlow; a backend's module is imported the first time it is requested.

Built-in propagators:

========================  ===================================================
``"numpy"``               Symmetric split-step Fourier (scalar NLSE /
                          Manakov) in NumPy. Always available; reference
                          implementation; default for :class:`~hpcom.Fiber`.
``"numpy-linear"``        Exact linear solution (loss + dispersion, γ
                          ignored) in one FFT. For linear-regime checks.
``"jax"``                 Every scheme on JAX/XLA (GPU/TPU if present).
                          Requires ``pip install "hpcom[jax]"``
                          (``"hpcom[jax-cuda]"`` for NVIDIA GPUs).
``"tensorflow"``          Every scheme on TensorFlow (GPU if present).
                          Requires ``pip install "hpcom[tf]"``.
========================  ===================================================

Adding a backend: subclass :class:`Propagator`, then
``register_propagator("name", MyPropagator)``. Every registered propagator is
exercised by the conformance suite in ``tests/backends/``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from hpcom.backends.base import BackendUnavailableError, CPUFallbackWarning, Propagator

__all__ = [
    "BackendUnavailableError",
    "CPUFallbackWarning",
    "Propagator",
    "available_propagators",
    "get_propagator",
    "list_propagators",
    "register_propagator",
]

#: name → (module, class name, install hint shown when the import fails)
_BUILTIN: dict[str, tuple[str, str, str]] = {
    "numpy": ("hpcom.backends.numpy.ssfm", "NumpySSFMPropagator", ""),
    "numpy-linear": ("hpcom.backends.numpy.linear", "LinearNumpyPropagator", ""),
    "jax": (
        "hpcom.backends.jax.propagator",
        "JaxPropagator",
        'install JAX with `pip install "hpcom[jax]"` (NVIDIA GPU: "hpcom[jax-cuda]")',
    ),
    "tensorflow": (
        "hpcom.backends.tensorflow.propagator",
        "TensorFlowPropagator",
        'install TensorFlow with `pip install "hpcom[tf]"` (or "hpcom[tf-gpu]")',
    ),
}

_factories: dict[str, Callable[[], Propagator]] = {}
_instances: dict[str, Propagator] = {}


def _builtin_factory(module: str, cls_name: str, hint: str, name: str) -> Callable[[], Propagator]:
    def factory() -> Propagator:
        try:
            cls = getattr(importlib.import_module(module), cls_name)
        except (ImportError, OSError) as exc:
            # OSError: broken CUDA libraries can fail inside `import tensorflow`.
            msg = f"propagator backend {name!r} is unavailable: {exc}"
            raise BackendUnavailableError(f"{msg}. To enable it, {hint}." if hint else msg) from exc
        instance = cls()
        assert isinstance(instance, Propagator)
        return instance

    return factory


def register_propagator(
    name: str, propagator: Propagator | type[Propagator] | Callable[[], Propagator]
) -> None:
    """Register ``propagator`` under ``name`` (replacing any previous entry).

    ``propagator`` may be an instance, a :class:`Propagator` subclass
    (instantiated lazily with no arguments), or a zero-argument factory.
    """
    if isinstance(propagator, Propagator):
        instance = propagator
        _factories[name] = lambda: instance
    elif isinstance(propagator, type):
        if not issubclass(propagator, Propagator):
            raise TypeError(f"{propagator.__name__} is not a Propagator subclass")
        _factories[name] = propagator
    elif callable(propagator):
        _factories[name] = propagator
    else:
        raise TypeError(
            "propagator must be a Propagator instance, subclass, or zero-argument factory; "
            f"got {type(propagator).__name__}"
        )
    _instances.pop(name, None)


def get_propagator(name: str) -> Propagator:
    """Return the propagator registered as ``name``, importing it on first use.

    Raises
    ------
    KeyError
        No backend of that name is registered.
    BackendUnavailableError
        The backend is registered but its dependencies are not installed.
    """
    if name in _instances:
        return _instances[name]
    if name not in _factories:
        raise KeyError(
            f"Unknown propagator backend {name!r}. Registered: {', '.join(list_propagators())}."
        )
    instance = _factories[name]()
    if not isinstance(instance, Propagator):
        raise TypeError(f"factory for backend {name!r} returned {type(instance).__name__}")
    _instances[name] = instance
    return instance


def list_propagators() -> list[str]:
    """Sorted names of every registered backend (available or not)."""
    return sorted(_factories)


def available_propagators() -> list[str]:
    """Sorted names of registered backends whose dependencies import successfully."""
    names = []
    for name in list_propagators():
        try:
            get_propagator(name)
        except BackendUnavailableError:
            continue
        names.append(name)
    return names


for _name, (_module, _cls, _hint) in _BUILTIN.items():
    _factories[_name] = _builtin_factory(_module, _cls, _hint, _name)
del _name, _module, _cls, _hint
