# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from harlow import __version__  # noqa: E402

sys.path.insert(0, os.path.abspath("../../harlow"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../"))
now = datetime.datetime.now()

# -- Project information -----------------------------------------------------

project = "Adaptive sampling"
copyright = f"{now.year}, TNO"
author = "TNO"
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    # "sphinx_rtd_dark_mode",
    "autoapi.extension",
]

# Autosummary
autosummary_generate = False
autoapi_dirs = ["../../harlow"]

# Sphinx gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "download_all_examples": False,
    "show_signature": False,
    "remove_config_comments": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "_templates"]

# bibliography settings
bibtex_bibfiles = ["harlow_references.bib"]
bibtex_reference_style = "author_year"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# # Rtd dark mode settings
# default_dark_mode = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["./_static"]

# this adds a custom javascript file that
# * opens all external link in a new tab
html_js_files = ["js/custom.js"]
