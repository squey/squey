kind: cmake
description: |

    A tool for use with clang to analyze includes in C and C++ source files

sources:
- kind: tar

  url: github:include-what-you-use/include-what-you-use/archive/refs/tags/0.22.tar.gz
  ref: 34c7636da2abe7b86580b53b762f5269e71efff460f24f17d5913c56eb99cb7c

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