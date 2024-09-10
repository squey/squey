kind: meson
description: |

    NumPy is the fundamental package needed for scientific computing with Python.

sources:
- kind: git
  url: github:numpy/numpy
  ref: 48606ab22bfdb0e9d7ec4ed5eef984b873b7796d
    
depends:
- base.bst
- lapack.bst

build-depends:
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/python3-pip.bst
- freedesktop-sdk.bst:components/cython.bst
- freedesktop-sdk.bst:components/git.bst
- meson.bst # using the numpy fork that contains the required "features" module
 
