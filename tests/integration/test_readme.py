"""The code in README.md and the documentation's quick start runs as written."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_RST_BLOCK = re.compile(r"\.\. code-block:: python\n\n((?:   .*\n|\n)+)")


def _readme_blocks() -> list[str]:
    return _BLOCK.findall((ROOT / "README.md").read_text())


def _index_blocks() -> list[str]:
    blocks = _RST_BLOCK.findall((ROOT / "docs" / "index.rst").read_text())
    return ["\n".join(line[3:] for line in b.splitlines()) for b in blocks]


@pytest.mark.slow
@pytest.mark.parametrize("code", _readme_blocks() + _index_blocks())
def test_documented_code_runs(code: str):
    exec(compile(code, "<documentation>", "exec"), {})


def test_documentation_has_code_blocks():
    assert _readme_blocks() and _index_blocks()
