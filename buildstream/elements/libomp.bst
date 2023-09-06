kind: cmake
description: |

    LLVM OpenMP library implementation 

sources:
- kind: tar
  url: https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/openmp-16.0.6.src.tar.xz
  ref: a2536f06373774f2829d892b72eea5f97bab20e69b26b03042d947f992eb8e38

- kind: patch
  path: buildstream/patches/libomp/ExtendPath.cmake.patch
  strip-level: 1

variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release

depends:
- base.bst
- freedesktop-sdk.bst:components/llvm.bst
