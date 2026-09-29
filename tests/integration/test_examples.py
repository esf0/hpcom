"""Every script in ``examples/`` runs to completion (documentation must not rot)."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[2] / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    if "hpcom.autodiff" in script.read_text() and importlib.util.find_spec("jax") is None:
        pytest.skip("example needs JAX")
    monkeypatch.setattr(sys, "argv", [str(script), str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_there_are_examples():
    assert len(EXAMPLES) >= 10
