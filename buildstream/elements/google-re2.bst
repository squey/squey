kind: cmake
description: |

    RE2 is a fast, safe, thread-friendly alternative to backtracking regular expression engines like those used in PCRE, Perl, and Python. It is a C++ library.
    
sources:
- kind: tar
  url: github:google/re2/archive/refs/tags/2024-07-02.tar.gz
  ref: eb2df807c781601c14a260a507a5bb4509be1ee626024cb45acbd57cb9d4032b

(@): buildstream/elements/base/target.bst

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
