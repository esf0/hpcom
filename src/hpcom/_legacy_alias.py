"""Lazy aliases that keep pre-v1 import paths working for one release.

Pre-v1 code imported modules such as ``hpcom.channel`` or ``hpcom.signal``.
Those now live in :mod:`hpcom._legacy`. This module installs an import hook
so that ``import hpcom.channel`` / ``from hpcom.signal import generate_wdm``
resolve to the legacy module *on first use* — ``import hpcom`` itself never
imports the legacy code (which pulls in TensorFlow and scikit-learn) and
never emits a warning. The :class:`DeprecationWarning` is raised by
:mod:`hpcom._legacy` when a legacy module is actually loaded.

Only names that the v1 package does not own are aliased: ``hpcom.metrics``
and ``hpcom.modulation`` are v1 modules; their legacy versions are reachable
as ``hpcom._legacy.metrics`` / ``hpcom._legacy.modulation``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType

#: Legacy submodules exposed under their pre-v1 ``hpcom.<name>`` path.
LEGACY_ALIASES: tuple[str, ...] = ("channel", "signal", "plot", "decorators", "hpcom_old")

_PACKAGE = "hpcom"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(self._target)

    def exec_module(self, module: ModuleType) -> None:
        """The target module is already executed by ``create_module``."""


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        package, _, name = fullname.rpartition(".")
        if package != _PACKAGE or name not in LEGACY_ALIASES:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(f"{_PACKAGE}._legacy.{name}"))


def install() -> None:
    """Register the alias finder (idempotent)."""
    if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        sys.meta_path.append(_LegacyAliasFinder())


def load(name: str) -> ModuleType:
    """Import legacy module ``name`` through its ``hpcom.<name>`` alias."""
    return importlib.import_module(f"{_PACKAGE}.{name}")
