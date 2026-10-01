"""Sphinx configuration for the neural-encoder documentation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_encoder import __version__  # noqa: E402


project = "neural-encoder"
author = "Pablo Marcos-Manchón"
copyright = "2026, Pablo Marcos-Manchón"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

html_theme = "furo"
html_title = "neural-encoder"
html_static_path = ["assets"]
html_theme_options = {
    "light_logo": "neural-encoder.svg",
    "dark_logo": "neural-encoder-dark.svg",
    "source_repository": "https://github.com/pablomm/neural-encoder/",
    "source_branch": "main",
    "source_directory": "docs/",
}
