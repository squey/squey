kind: meson
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

sources:
- kind: git
  url: github:numpy/numpy
  ref: f2a77a76e08719556527e0819182073fe9b5f1c3

depends:
- base.bst
- lapack.bst

build-depends:
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/python3-pip.bst
- freedesktop-sdk.bst:components/cython.bst
- freedesktop-sdk.bst:components/git.bst
- meson.bst # using the numpy fork that contains the required "features" module
