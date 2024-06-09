kind: cmake
description: |

    A tool for use with clang to analyze includes in C and C++ source files

sources:
- kind: tar

  url: github:include-what-you-use/include-what-you-use/archive/refs/tags/0.21.tar.gz
  ref: a472fe8587376d041585c72e5643200f8929899f787725f0ba9e5b3d3820d401

variables:
  cmake-local: >-
    -DCMAKE_PREFIX_PATH=/usr
    -DCMAKE_INSTALL_RPATH="/usr/lib"

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/libxml2.bst
- freedesktop-sdk.bst:components/zstd.bst
- llvm.bst