"""Pytest configuration shared across the test tree.

* ``@pytest.mark.gpu`` tests run only with ``--rungpu`` (or ``HPCOM_RUN_GPU=1``).
* ``nonlinear_backend`` / ``any_backend`` parametrise a test over every
  *available* propagator, so a newly registered backend (JAX, Rust, …) is
  automatically held to the same physics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # make `helpers` importable

from hpcom.backends import available_propagators

#: Backends that solve the full NLSE / Manakov equation.
NONLINEAR_BACKENDS = [name for name in available_propagators() if name != "numpy-linear"]
ALL_BACKENDS = available_propagators()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--rungpu",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.gpu (skipped by default)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--rungpu") or os.environ.get("HPCOM_RUN_GPU") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="gpu test: pass --rungpu to enable")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(params=NONLINEAR_BACKENDS)
def nonlinear_backend(request: pytest.FixtureRequest) -> str:
    """Name of every available backend that solves the nonlinear equation."""
    return str(request.param)


@pytest.fixture(params=ALL_BACKENDS)
def any_backend(request: pytest.FixtureRequest) -> str:
    """Name of every available backend (including the linear-only one)."""
    return str(request.param)
