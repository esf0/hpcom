"""Sphinx configuration for hpcom."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version

# Make ``hpcom`` importable when running ``sphinx-build`` directly from a
# checkout (without installing). Both src/ layout and editable installs work.
sys.path.insert(0, os.path.abspath("../src"))

project = "hpcom"
author = "Egor Sedov"
copyright = "2026, Egor Sedov"  # noqa: A001 — Sphinx convention

try:
    release = _pkg_version("hpcom")
except Exception:  # noqa: BLE001 — sphinx must build even if metadata is missing
    release = "0.2.0.dev0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
]

source_suffix = [".rst"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "intent", "spec", "source"]

# Napoleon: NumPy-style docstrings.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Mock the heavy / optional deps so the docs build doesn't pull TF on RTD.
autodoc_mock_imports = ["tensorflow", "ssfm_gpu"]
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"exclude-members": "model_config, model_fields, model_computed_fields"}
# pydantic models render their fields from the docstrings; field signatures are noise.
autodoc_class_signature = "separated"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = f"hpcom {release}"
html_static_path: list[str] = []
