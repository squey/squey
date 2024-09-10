kind: cmake
description: |

    RE2 is a fast, safe, thread-friendly alternative to backtracking regular expression engines like those used in PCRE, Perl, and Python. It is a C++ library.
    
sources:
- kind: tar
  url: github:google/re2/archive/refs/tags/2024-07-02.tar.gz
  ref: eb2df807c781601c14a260a507a5bb4509be1ee626024cb45acbd57cb9d4032b

variables:
  cmake-local: |
    -DCMAKE_CXX_FLAGS="-fPIC"

depends:
- base.bst
- freedesktop-sdk.bst:components/abseil-cpp.bst
