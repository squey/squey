kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: github:pybind/pybind11/archive/refs/tags/v3.1.0.tar.gz
  ref: ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665

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
