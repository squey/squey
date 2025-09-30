kind: cmake
description: |

    utf8proc is a small, clean C library that provides Unicode normalization, case-folding, and other operations for data in the UTF-8 encoding.

sources:
- kind: tar
  url: github:JuliaStrings/utf8proc/archive/refs/tags/v2.10.0.tar.gz
  ref: 6f4f1b639daa6dca9f80bc5db1233e9cbaa31a67790887106160b33ef743f136

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
