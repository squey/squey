kind: cmake
description: |

    utf8proc is a small, clean C library that provides Unicode normalization, case-folding, and other operations for data in the UTF-8 encoding.

sources:
- kind: tar
  url: github:JuliaStrings/utf8proc/archive/refs/tags/v2.11.3.tar.gz
  ref: abfed50b6d4da51345713661370290f4f4747263ee73dc90356299dfc7990c78

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
