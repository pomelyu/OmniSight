"""Sphinx configuration for OmniSight documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project info -------------------------------------------------------------

project = "OmniSight"
copyright = "2024, OmniSight Contributors"
author = "OmniSight Contributors"
release = "0.1.0"

# -- Extensions ---------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# -- autodoc ------------------------------------------------------------------

# Mock heavy runtime dependencies so Sphinx can import the package without
# needing them installed in the docs build environment.
autodoc_mock_imports = ["cv2", "numpy", "onnxruntime"]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "private-members": False,
}

# -- autosummary --------------------------------------------------------------

autosummary_generate = True

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_returns = False
napoleon_use_rtype = True

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = "OmniSight"
