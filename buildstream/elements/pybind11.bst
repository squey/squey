kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: github:pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
  ref: 741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca

(@): buildstream/elements/base/target.yml

environment:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      Python_ROOT_DIR: "%{prefix}"

variables:
  cmake-local: >-
    -DPYBIND11_TEST=0
    -DPYBIND11_NOPYTHON=ON

build-depends:
- base.bst
- boost.bst

(?):
- (target_triple == "x86_64-linux-gnu"):
    build-depends:
      (>):
      - freedesktop-sdk.bst:components/python3.bst
- (target_triple != "x86_64-linux-gnu"):
    build-depends:
      (>):
      - python3_bin.bst
      - freedesktop-sdk.bst:components/python3.bst
