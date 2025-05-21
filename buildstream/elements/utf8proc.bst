kind: cmake
description: |

    utf8proc is a small, clean C library that provides Unicode normalization, case-folding, and other operations for data in the UTF-8 encoding.
    
sources:
- kind: tar
  url: github:JuliaStrings/utf8proc/archive/refs/tags/v2.9.0.tar.gz
  ref: 18c1626e9fc5a2e192311e36b3010bfc698078f692888940f1fa150547abb0c1

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON

(@): buildstream/elements/base/target.bst

depends:
- sdk.bst
