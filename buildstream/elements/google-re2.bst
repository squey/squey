kind: cmake
description: |

    RE2 is a fast, safe, thread-friendly alternative to backtracking regular expression engines like those used in PCRE, Perl, and Python. It is a C++ library.

sources:
- kind: tar
  url: github:google/re2/archive/refs/tags/2025-08-12.tar.gz
  ref: 2f3bec634c3e51ea1faf0d441e0a8718b73ef758d7020175ed7e352df3f6ae12

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
