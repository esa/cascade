# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cascade'
copyright = '2023, Francesco Biscani and Dario Izzo'
author = 'Francesco Biscani and Dario Izzo'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_nb", "sphinx.ext.intersphinx"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/esa/cascade",
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
    "*Trappist-1*",
    "*Outer*",
    "*Maxwell*",
    "*Keplerian billiard*",
    "*embryos*",
    "tides_spokes*",
    "ensemble_batch_perf*",
    "The restricted three-body problem*",
    "parallel_mode.ipynb",
    "vsop2013.ipynb",
    "compiled_functions.ipynb",
]

latex_engine = "xelatex"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]