kind: cmake
description: |

    RE2 is a fast, safe, thread-friendly alternative to backtracking regular expression engines like those used in PCRE, Perl, and Python. It is a C++ library.

sources:
- kind: tar
  url: github:google/re2/archive/refs/tags/2025-11-05.tar.gz
  ref: 87f6029d2f6de8aa023654240a03ada90e876ce9a4676e258dd01ea4c26ffd67

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DCMAKE_CXX_FLAGS="-fPIC"
    -DBUILD_SHARED_LIBS=ON

build-depends:
  - sdk.bst

(?):
  - (target_triple != "x86_64-linux-gnu"):
      depends:
        - abseil-cpp.bst
  - target_triple == "x86_64-linux-gnu":
      depends:
        - freedesktop-sdk.bst:components/abseil-cpp.bst
