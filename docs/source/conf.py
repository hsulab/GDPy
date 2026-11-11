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
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "gdpx"
copyright = '2020-2026, Jiayan Xu'
author = 'Jiayan Xu'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_static/**', '_templates/**']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "font-stack": '"gdpx JetBrains Mono", monospace',
        "font-stack--headings": '"gdpx JetBrains Mono", monospace',
        "font-stack--monospace": '"gdpx JetBrains Mono", monospace',
        "sidebar-item-font-size": "0.875rem",
        "sidebar-caption-font-size": "0.875rem",
        "sidebar-search-input-font-size": "0.875rem",
        "sidebar-item-line-height": "1.4rem",
        "sidebar-item-spacing-vertical": "0.25rem",
        "sidebar-caption-space-above": "0.75rem",
        "sidebar-tree-space-above": "0.75rem",
        "toc-font-size": "1rem",
        "toc-font-size--mobile": "1rem",
        "toc-title-font-size": "1rem",
        "code-font-size": "0.875rem",
        "admonition-font-size": "0.875rem",
        "admonition-title-font-size": "0.875rem",
        "api-font-size": "0.875rem",
        "color-nav-command": "#246653",
        "color-nav-secondary": "#985715",
    },
    "dark_css_variables": {
        "color-nav-command": "#8fd3c1",
        "color-nav-secondary": "#f0ad66",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/typography.css', 'css/navigation.css']
