"""Pytest configuration shared across the test tree.

Gates ``@pytest.mark.gpu`` tests behind an explicit ``--rungpu`` flag (or
``HPCOM_RUN_GPU=1`` env var). Default CI runs skip them so the standard suite
stays portable; self-hosted GPU runners opt in with ``pytest --rungpu``.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--rungpu",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.gpu (skipped by default)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--rungpu") or os.environ.get("HPCOM_RUN_GPU") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="gpu test: pass --rungpu to enable")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
