kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

sources:
- kind: git
  url: https://github.com/numpy/numpy
  ref: 21cacafbeeff812cf49ca0102d1dd582b6d77e2b
  checkout-submodules: True
  
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
- freedesktop-sdk.bst:components/git.bst
 
