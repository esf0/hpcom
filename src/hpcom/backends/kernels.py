"""DSP kernel interface and registry — the seam for compiled (e.g. Rust) kernels.

hpcom's transmitter/receiver code never calls a hot loop directly; it calls
``get_kernels().<kernel>(...)``. A kernel set is any object providing the
functions of :class:`DSPKernels` with identical semantics (bit-exact
outputs). The pure-NumPy set is registered as ``"python"`` and is the
default; a compiled set is added with::

    from hpcom.backends.kernels import register_kernels, set_default_kernels
    register_kernels("rust", my_rust_module)   # module or object
    set_default_kernels("rust")

Kernel choice changes speed only, never results — the conformance suite
(``tests/backends/test_kernel_conformance.py``) runs every registered set
against the same expectations.
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "DSPKernels",
    "get_kernels",
    "list_kernels",
    "register_kernels",
    "set_default_kernels",
]


@runtime_checkable
class DSPKernels(Protocol):
    """Functions every kernel set provides (see :mod:`hpcom.backends.python.kernels`)."""

    def nearest_indices(
        self, received: npt.NDArray[np.complex128], points: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.intp]: ...

    def ccdm_encode(
        self,
        bits: npt.NDArray[np.uint8],
        composition: npt.NDArray[np.int64],
        bits_per_block: int,
    ) -> npt.NDArray[np.intp]: ...

    def ccdm_decode(
        self,
        indices: npt.NDArray[np.intp],
        composition: npt.NDArray[np.int64],
        bits_per_block: int,
    ) -> npt.NDArray[np.uint8]: ...


_registry: dict[str, object] = {}
_default = "python"


def register_kernels(name: str, kernels: object) -> None:
    """Register a kernel set (a module or object implementing :class:`DSPKernels`)."""
    missing = [
        fn
        for fn in ("nearest_indices", "ccdm_encode", "ccdm_decode")
        if not callable(getattr(kernels, fn, None))
    ]
    if missing:
        raise TypeError(f"kernel set {name!r} is missing {missing}")
    _registry[name] = kernels


def get_kernels(name: str | None = None) -> DSPKernels:
    """Return kernel set ``name`` (default: the one chosen by :func:`set_default_kernels`)."""
    key = _default if name is None else name
    if key == "python" and key not in _registry:
        register_kernels("python", importlib.import_module("hpcom.backends.python.kernels"))
    try:
        kernels = _registry[key]
    except KeyError:
        raise KeyError(f"unknown kernel set {key!r}; registered: {list_kernels()}") from None
    return kernels  # type: ignore[return-value]  # structurally checked in register_kernels


def list_kernels() -> list[str]:
    """Names of registered kernel sets."""
    get_kernels("python")
    return sorted(_registry)


def set_default_kernels(name: str) -> None:
    """Select the kernel set used when :func:`get_kernels` is called without a name."""
    global _default
    get_kernels(name)  # validate
    _default = name
