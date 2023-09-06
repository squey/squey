kind: cmake
description: |

    The Open Standard for Parallel Programming of Heterogeneous Systems

sources:
- kind: tar
  url: https://codeload.github.com/pocl/pocl/tar.gz/refs/tags/v4.0
  ref: 7f4e8ab608b3191c2b21e3f13c193f1344b40aba7738f78762f7b88f45e8ce03

variables:
  cmake-local: |
    -DINSTALL_OPENCL_HEADERS=1 \
    -DEXTRA_HOST_LD_FLAGS=-L%{prefix}/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLC_HOST_CPU=corei7

depends:
- base.bst
- freedesktop-sdk.bst:components/llvm.bst
- freedesktop-sdk.bst:components/opencl.bst
- freedesktop-sdk.bst:components/patch.bst
- freedesktop-sdk.bst:components/python3.bst
- libhwloc.bst
