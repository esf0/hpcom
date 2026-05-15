"""Legacy hpcom modules (pre-v1).

Preserved for one release to ease migration. Will be removed in v2.
New code should use the public API at ``hpcom.*`` (see ``docs/spec/hpcom-v1.md``).

Submodules are imported best-effort: if a legacy module fails to import (e.g.
because an external dep like ``ssfm-gpu`` is missing or has drifted), it is
skipped here rather than breaking the whole package. The set of successfully
imported submodules is recorded in ``__all__``.
"""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "hpcom._legacy is deprecated and will be removed in v2. "
    "Migrate to the new hpcom public API; see docs/spec/hpcom-v1.md.",
    DeprecationWarning,
    stacklevel=2,
)

_CANDIDATE_SUBMODULES = (
    "decorators",
    "modulation",
    "metrics",
    "plot",
    "signal",
    "channel",
    "hpcom_old",
    "_sklearn_functions",
)

__all__: list[str] = []

for _name in _CANDIDATE_SUBMODULES:
    try:
        _mod = importlib.import_module(f"{__name__}.{_name}")
    except Exception as _exc:  # noqa: BLE001 — legacy may break in arbitrary ways
        warnings.warn(
            f"hpcom._legacy.{_name} failed to import and is unavailable: {_exc!r}",
            ImportWarning,
            stacklevel=2,
        )
        continue
    globals()[_name] = _mod
    __all__.append(_name)

del _CANDIDATE_SUBMODULES, _name
