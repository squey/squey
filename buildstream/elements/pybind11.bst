kind: cmake
description: |

   Seamless operability between C++11 and Python

sources:
- kind: tar
  url: github:pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
  ref: e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20

(@): buildstream/elements/base/target.bst

environment:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      Python_ROOT_DIR: "%{bindir}/python"

variables:
  cmake-local: >-
    -DPYBIND11_TEST=0

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