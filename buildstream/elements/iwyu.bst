kind: cmake
description: |

    A tool for use with clang to analyze includes in C and C++ source files

sources:
- kind: tar

  url: github:include-what-you-use/include-what-you-use/archive/refs/tags/0.26.tar.gz
  ref: ca3f157a5e6bf293cfb2fea585b103ec87245d02eaee0b6ab832665e740016eb

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
