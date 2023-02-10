kind: cmake
description: |

    LLVM OpenMP library implementation 

sources:
- kind: tar
  url: https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/openmp-14.0.6.src.tar.xz
  ref: 4f731ff202add030d9d68d4c6daabd91d3aeed9812e6a5b4968815cfdff0eb1f
  base-dir: 'openmp-14.0.6.src'

variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release

depends:
- base.bst
- freedesktop-sdk.bst:components/llvm.bst
