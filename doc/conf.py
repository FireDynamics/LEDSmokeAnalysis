
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version as get_version
try:
    import tomllib  # Python 3.11 and newer
except ModuleNotFoundError:
    tomllib = None
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LEDSmokeAnalysis'
copyright = '2026, CCE'
author = 'CCE'
# The version of the source tree these docs document, as its pyproject.toml states it. The metadata of an
# installed ledsa can be older: an editable install keeps the version it was installed with.
if tomllib is not None:
    with open(os.path.join(os.path.abspath('..'), 'pyproject.toml'), 'rb') as f:
        release = tomllib.load(f)['project']['version']
else:
    release = get_version('ledsa')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_copybutton'
]

autodoc_mock_imports = [
    "pandas",
    "matplotlib",
    "tqdm",
    "scipy",
    "exifread",
    "rawpy",
    "opencv-python",
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['tables.css']