# Copyright 2026 The HRAP Authors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Based on https://github.com/jax-ml/jax/blob/main/docs/conf.py

project = 'hrap'
copyright = '2026, The HRAP Authors'
author = 'Thomas A. Scott, Drew Nickel'
release = ''
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

needs_sphinx = '2.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# import sys, os
# sys.path.append(os.path.abspath('sphinxext'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
    "sphinx_remove_toctrees",
    'sphinx_copybutton',
#    'jax_extensions',
#    'jax_list_config_options',
    'sphinx_design',
    # 'sphinxext.rediraffe',
    # 'source_include',
    'sphinxcontrib.mermaid'
]
templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # Sometimes sphinx reads its own outputs as inputs!
    'build/html',
    'build/jupyter_execute',
    'notebooks/README.md',
    'README.md',
    # Ignore markdown source for notebooks; myst-nb builds from the ipynb
    # These are kept in sync using the jupytext pre-commit hook.
    'notebooks/*.md',
    'pallas/quickstart.md',
    'pallas/pipelining.md',
    'pallas/gpu/pipelining.md',
    'pallas/tpu/pipelining.md',
    'pallas/tpu/distributed.md',
    'pallas/tpu/sparse.md',
    'pallas/tpu/matmul.md',
    'pallas/tpu/core_map.md',
    'jep/9407-type-promotion.md',
    'autodidax.md',
    'autodidax2_part1.md',
    'array_refs.md',
    'sharded-computation.md',
    'ffi.ipynb',
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/rnickel1/HRAP_Source',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
    'article_header_start': ['toggle-primary-sidebar.html', 'breadcrumbs'],
}

# html_logo = '_static/jax_logo_250px.png'
# html_favicon = '_static/favicon.png'

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath']
myst_ref_domains = ["py"]
myst_all_links_external = False
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_show_tb = True

autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_type_aliases = {
    'ArrayLike': 'jax.typing.ArrayLike',
    'DTypeLike': 'jax.typing.DTypeLike',
}

# # Use order it was defined in the source
# autodoc_member_order = 'bysource'

# Remove auto-generated API docs from sidebars
remove_from_toctrees = ["_autosummary/*"]

# Customize code links via sphinx.ext.linkcode
def linkcode_resolve(domain, info):
  import jax

  if domain != 'py':
    return None
  if not info['module']:
    return None
  if not info['fullname']:
    return None
  if info['module'].split(".")[0] != 'jax':
     return None
  try:
    mod = sys.modules.get(info['module'])
    obj = operator.attrgetter(info['fullname'])(mod)
    if isinstance(obj, property):
        obj = obj.fget
    while hasattr(obj, '__wrapped__'):  # decorated functions
        obj = obj.__wrapped__
    filename = inspect.getsourcefile(obj)
    source, linenum = inspect.getsourcelines(obj)
  except:
    return None
  try:
    filename = Path(filename).relative_to(Path(jax.__file__).parent)
  except ValueError:
    # Source file is not a relative to jax; this must be a re-exported function.
    return None
  lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""
  return f"https://github.com/rnickel1/HRAP_Source/blob/main/HRAP%20-%20Python/{filename}{lines}"
