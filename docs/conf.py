# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "AmorphGen"
copyright = "2026, Chaiyawat Kaewmeechai"
author = "Chaiyawat Kaewmeechai"
version = "1.0.0rc2"
release = "1.0.0rc2"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",                # Markdown support
    "sphinx.ext.autodoc",         # Auto-generate from docstrings
    "sphinx.ext.autosummary",     # Summary tables for modules
    "sphinx.ext.napoleon",        # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",        # [source] links to highlighted code
    "sphinx.ext.intersphinx",     # Cross-link to ASE, NumPy, etc.
    "sphinx_copybutton",          # Copy button on code blocks
    "sphinx_design",              # Tabs, cards, grids
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",       # ::: directive syntax
    "deflist",           # Definition lists
    "fieldlist",         # Field lists
    "substitution",      # |variable| substitution
    "tasklist",          # - [ ] checkboxes
]
myst_heading_anchors = 3

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "ase",
    "mace",
    "mace_torch",
    "chgnet",
    "sevenn",
    "torch",
    "numpy",
]

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True

# -- Napoleon configuration ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration ------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# -- HTML output options ------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"
html_title = "AmorphGen"

# -- Source suffix ------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
