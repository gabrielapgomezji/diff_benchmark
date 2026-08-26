# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DiffBench'
copyright = '2026, Gabriela Gomez Jimenez'
author = 'Gabriela Gomez Jimenez'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.githubpages",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo' #'alabaster'
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "DiffBench",
    },
    "show_prev_next": True,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gabrielapgomezji/diff_benchmark",
            "icon": "fa-brands fa-github",
        }
    ],
}
html_static_path = ['_static']
html_css_files = [
    "custom.css",
]
