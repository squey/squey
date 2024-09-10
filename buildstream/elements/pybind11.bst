kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: github:pybind/pybind11/archive/refs/tags/v2.13.5.tar.gz
  ref: b1e209c42b3a9ed74da3e0b25a4f4cd478d89d5efbb48f04b277df427faf6252
  
variables:
  cmake-local: |
    -DPYBIND11_TEST=0
    
build-depends:
- base.bst
- boost.bst
- freedesktop-sdk.bst:components/python3.bst
