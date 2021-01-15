# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Good resources:
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html#theme-options
# https://docs.readthedocs.io/en/latest/guides/adding-custom-css.html

import os

import sys
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../../../src'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
for x in os.walk('../../src'):
    sys.path.insert(0, x[0])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
source_suffix = '.rst'
master_doc = 'index'
project = 'menten_gcn'
year = '2020'
author = 'Menten AI, Inc.'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.1.0'

pygments_style = 'trac'
templates_path = ['_templates']
extlinks = {
    'issue': ('https://github.com/MentenAI/menten_gcn/issues/%s', '#'),
    'pr': ('https://github.com/MentenAI/menten_gcn/pull/%s', 'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
html_logo = "_static/white_logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'logo_url': "https://menten.ai/"
}

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
