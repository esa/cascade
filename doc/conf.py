# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cascade'
copyright = '2023, Francesco Biscani and Dario Izzo'
author = 'Francesco Biscani and Dario Izzo'

# The full version, including alpha/beta/rc tags
import cascade as csc

release = csc.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_nb", "sphinx.ext.intersphinx"]

intersphinx_mapping = {
    "hy": ("https://bluescarni.github.io/heyoka.py", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/images/logo.png"

html_theme_options = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/bluescarni/heyoka.py",
    "repository_branch": "main",
    "path_to_docs": "doc",
    "use_repository_button": True,
    "use_issues_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
}

nb_execution_mode = "force"

nb_execution_excludepatterns = [
    "leo_population.ipynb",
    "simple_atmosphere.ipynb",
    "*cubes.ipynb"
]

latex_engine = "xelatex"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]