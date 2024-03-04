kind: cmake
description: |

    RE2 is a fast, safe, thread-friendly alternative to backtracking regular expression engines like those used in PCRE, Perl, and Python. It is a C++ library.
    
sources:
- kind: tar
  url: github:google/re2/archive/refs/tags/2024-03-01.tar.gz
  ref: 7b2b3aa8241eac25f674e5b5b2e23d4ac4f0a8891418a2661869f736f03f57f4

variables:
  cmake-local: |
    -DCMAKE_CXX_FLAGS="-fPIC"

depends:
- base.bst
- abseil.bst
