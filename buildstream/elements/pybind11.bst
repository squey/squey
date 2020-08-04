kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz
  ref: cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a
  
variables:
  cmake-local: |
    -DPYBIND11_TEST=0
    
build-depends:
- base.bst
- boost.bst
- freedesktop-sdk.bst:components/python3.bst
