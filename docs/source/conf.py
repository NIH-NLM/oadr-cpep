import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'oadr-cpep'
copyright = '2026, NIH-NLM'
author = 'NIH-NLM'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]

# Mock imports for packages not available during doc build
autodoc_mock_imports = [
    'numpy',
    'pandas',
    'sklearn',
    'scipy',
    'matplotlib',
    'typer',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
