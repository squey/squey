kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

sources:
- kind: tar
  url: https://codeload.github.com/numpy/numpy/tar.gz/v1.19.4
  ref: 6082884371a5998fa8b006b506f4827c0617d789d7b3ee78549bb044139a9a8e
  
config:
  configure-commands:
  - |
    cat <<EOF >site.cfg
    [DEFAULT]
    library_dirs = /app/lib:/usr/local/lib64:/usr/local/lib:/usr/lib64:/usr/lib
    include_dirs = /app/include:/usr/local/include:/usr/include
    EOF
  install-commands:
    - python3 setup.py build -j `nproc` install --root=%{install-root} --prefix=%{prefix}

depends:
- base.bst
- lapack.bst
- freedesktop-sdk.bst:components/python3.bst

build-depends:
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/cython.bst
 
