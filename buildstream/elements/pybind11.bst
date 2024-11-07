kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: github:pybind/pybind11/archive/refs/tags/v2.13.5.tar.gz
  ref: b1e209c42b3a9ed74da3e0b25a4f4cd478d89d5efbb48f04b277df427faf6252

(@): buildstream/elements/base/target.bst
  
variables:
  cmake-local: |
    -DPYBIND11_TEST=0
    
build-depends:
- base.bst
- boost.bst

(?):
- (target_triple == "x86_64-linux-gnu"):
    build-depends:
      (>):
      - freedesktop-sdk.bst:components/python3.bst
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    build-depends:
      (>):
      - python3_darwin.bst