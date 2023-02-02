import sys
import os

sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '../Schrodinger'))))

project = 'Schrodinger'
copyright = '2023, Schrodinger'
author = 'Group Schrodinger'
release = '0.0.1'
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"
