# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme
import sphinxcontrib.napoleon

extensions = [
    "sphinx_rtd_theme",
    'sphinx.ext.autosectionlabel',
    "sphinxcontrib.napoleon",
    "rinoh.frontend.sphinx",
    "sphinx-prompt",
    'sphinx_substitution_extensions',
]

# -- Project information -----------------------------------------------------

# This will replace the |variables| within the rST documents automatically

OPENSOURCE_VERSION=True

if OPENSOURCE_VERSION:
    
    project = 'Open Federated Learning'
    author = 'FeTS'
    master_doc = 'index'
    
    # Global variables for rST
    rst_prolog = """
    .. |productName| replace:: Open Federated Learning
    .. |productZip| replace:: OpenFederatedLearning.zip
    .. |productDir| replace:: OpenFederatedLearning
    
    .. _Makefile: https://github.com/IntelLabs/OpenFederatedLearning/blob/master/Makefile
    """

    rinoh_documents = [('index', u'open_fl_manual', u'Open Federated Learning Manual', u'FeTS')]

else:
    
    project = 'FL.edge'
    copyright = '2020, Intel'
    author = 'Secure Intelligence Team'
    master_doc = 'index'
    
    # Global variables for rST
    rst_prolog = """
    .. |productName| replace:: Intel FL.Edge
    .. |productZip| replace:: intel_fledge.zip
    .. |productDir| replace:: intel_fledge
    
    .. _Makefile: https://github.com/IntelLabs/OpenFederatedLearning/blob/master/Makefile
    """

    rinoh_documents = [('index', u'intel_fledge_manual', u'Intel FL.edge manual', u'Intel')]


napoleon_google_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autosectionlabel_prefix_document = True
