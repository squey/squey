kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: https://github.com/pybind/pybind11/archive/v2.10.0.tar.gz
  ref: eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec
  
variables:
  cmake-local: |
    -DPYBIND11_TEST=0
    
build-depends:
- base.bst
- boost.bst
- freedesktop-sdk.bst:components/python3.bst
