"""The code in README.md, the cookbook and the documentation's quick start runs as written."""

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


RECIPES = sorted((ROOT / "cookbook").glob("[0-9][0-9]_*.md"))
_TEXT_BLOCK = re.compile(r"```text\n(.*?)```", re.DOTALL)


def _recipe(path: Path) -> tuple[str, str]:
    """The code of a recipe and the output it states (one block each)."""
    text = path.read_text()
    (code,), (output,) = _BLOCK.findall(text), _TEXT_BLOCK.findall(text)
    return code, output


def _documented():
    blocks = [(code, "") for code in _readme_blocks() + _index_blocks()]
    ids = [f"readme-{i}" for i in range(len(_readme_blocks()))]
    ids += [f"index-{i}" for i in range(len(_index_blocks()))]
    return blocks + [_recipe(p) for p in RECIPES], ids + [p.stem for p in RECIPES]


@pytest.mark.slow
@pytest.mark.parametrize(("code", "output"), _documented()[0], ids=_documented()[1])
def test_documented_code_runs(code: str, output: str, capsys: pytest.CaptureFixture[str]):
    if "import jax" in code or "learn" in code:  # hpcom.learn needs JAX
        pytest.importorskip("jax")
    exec(compile(code, "<documentation>", "exec"), {})
    printed = capsys.readouterr().out
    if output:  # a recipe prints its stated output (2 decimals of dB / 3 of bit/symbol)
        assert _numbers(printed) == pytest.approx(_numbers(output), rel=2e-3, abs=2e-2)


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", text)]


def test_documentation_has_code_blocks():
    assert _readme_blocks() and _index_blocks()
    assert len(RECIPES) >= 7
