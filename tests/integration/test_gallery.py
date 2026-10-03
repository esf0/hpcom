"""Every gallery entry runs in its quick mode (the reproductions must not rot)."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

ENTRIES = sorted((Path(__file__).parents[2] / "gallery").glob("*/run.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", ENTRIES, ids=[p.parent.name for p in ENTRIES])
def test_gallery_entry_runs(script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if importlib.util.find_spec("jax") is None:
        pytest.skip("the gallery needs JAX")
    monkeypatch.setattr(sys, "argv", [str(script), "--quick", "--out", str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    assert (tmp_path / "summary.md").read_text().count("|") > 10


def test_there_are_gallery_entries():
    assert len(ENTRIES) >= 3
