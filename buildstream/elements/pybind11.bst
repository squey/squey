kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: github:pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
  ref: d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
  
variables:
  cmake-local: |
    -DPYBIND11_TEST=0
    
build-depends:
- base.bst
- boost.bst
- freedesktop-sdk.bst:components/python3.bst
