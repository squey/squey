kind: cmake
description: |

    LLVM OpenMP library implementation

sources:
- kind: tar
  url: github:llvm/llvm-project/releases/download/llvmorg-19.1.7/openmp-19.1.7.src.tar.xz
  ref: bd7e6901ab086fd268750363017935fd4a717c153dad3c2aab86cb0140d9e3fe

- kind: tar
  url: github:llvm/llvm-project/releases/download/llvmorg-19.1.7/cmake-19.1.7.src.tar.xz
  ref: 11c5a28f90053b0c43d0dec3d0ad579347fc277199c005206b963c19aae514e3
  directory: cmake/

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DCMAKE_BUILD_TYPE=Release

config:
  configure-commands:
  - |
    cd cmake && for f in Modules/*.cmake; do ln -s "$f" .; done
    cd ..
    %{cmake}

depends:
- sdk.bst
- freedesktop-sdk.bst:components/llvm.bst
- freedesktop-sdk.bst:components/python3.bst
