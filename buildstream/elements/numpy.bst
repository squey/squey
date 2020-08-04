kind: manual
description: |

    NumPy is the fundamental package needed for scientific computing with Python.
    https://numpy.org/devdocs/release/1.18.1-notes.html

sources:
- kind: tar
  url: https://github.com/numpy/numpy/archive/v1.18.1.tar.gz
  ref: 96af6ec6c24e2df0d1591076a102b36bcc0c622411cbb191bca9a1ae724c4606
  
config:
  install-commands:
    - python3 setup.py build -j `nproc` install --root=%{install-root} --prefix=%{prefix}

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst

build-depends:
- freedesktop-sdk.bst:components/python3-setuptools.bst
- freedesktop-sdk.bst:components/cython.bst
 
