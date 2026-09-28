"""Tests of the pre-v1 modules under ``hpcom._legacy``.

They run only when the legacy dependencies (``pip install "hpcom[legacy]"``:
TensorFlow, scikit-learn, ssfm-gpu) are installed. Two files are scripts
rather than tests — they execute at import time — and are not collected:

* ``test.py`` prints GPU/TensorFlow diagnostics;
* ``test_propagation.py`` drives ``ssfm_gpu.conversion``, whose API has
  changed since the script was written (``KeyError: 'Q1'``).
"""

from __future__ import annotations

import importlib.util

import pytest

collect_ignore = ["test.py", "test_propagation.py"]

_MISSING = [m for m in ("tensorflow", "sklearn", "ssfm_gpu") if importlib.util.find_spec(m) is None]
if _MISSING:
    collect_ignore_glob = ["test_*.py"]

pytestmark = pytest.mark.legacy
