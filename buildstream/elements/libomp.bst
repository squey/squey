kind: cmake
description: |

    LLVM OpenMP library implementation 

sources:
- kind: tar
  url: github:llvm/llvm-project/releases/download/llvmorg-18.1.8/openmp-18.1.8.src.tar.xz
  ref: 60ed57245e73894e4a2a89b15889f367bd906abfe6d3f92e1718223d4b496150

- kind: patch
  path: buildstream/patches/libomp/ExtendPath.cmake.patch
  strip-level: 1

variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release

config:
  configure-commands:
  - |
    sed -i '20d;133,137d;148d;149d;' runtime/cmake/config-ix.cmake
    %{cmake}

depends:
- base.bst
- freedesktop-sdk.bst:components/llvm.bst
- freedesktop-sdk.bst:components/python3.bst
