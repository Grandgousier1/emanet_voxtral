#!/usr/bin/env python3
"""
Sphinx configuration for Voxtral documentation.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for autodoc
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
project = 'Emanet Voxtral'
copyright = '2025, Voxtral Team'
author = 'Voxtral Team'
version = '3.0'
release = '3.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'myst_parser',  # For Markdown support
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', '__pycache__']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_theme_options = {
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    'top_of_page_button': 'edit',
    'source_repository': 'https://github.com/your-org/voxtral/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Don't import modules during autodoc
autodoc_mock_imports = [
    'torch', 'transformers', 'vllm', 'librosa', 'soundfile', 
    'yt_dlp', 'rich', 'numpy'
]

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Custom configuration ----------------------------------------------------
# Enable source links
html_show_sourcelink = True
html_copy_source = True

# Add custom CSS
def setup(app):
    app.add_css_file('custom.css')