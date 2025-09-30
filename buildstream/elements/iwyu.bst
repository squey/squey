kind: cmake
description: |

    A tool for use with clang to analyze includes in C and C++ source files

sources:
- kind: tar

  url: github:include-what-you-use/include-what-you-use/archive/refs/tags/0.25.tar.gz
  ref: 2e8381368ec0a6ecb770834bce00fc62efa09a2b2f9710ed569acbb823ead9cc

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
