kind: cmake
description: |

    The Open Standard for Parallel Programming of Heterogeneous Systems

sources:
- kind: tar
  url: https://github.com/pocl/pocl/archive/v1.5.tar.gz
  ref: 4fcf4618171727d165fc044d465a66e3119217bb4577a97374f94fcd8aed330e
  
variables:
  cmake-local: |
    -DEXTRA_HOST_LD_FLAGS=-L%{prefix}/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLC_HOST_CPU=corei7

depends:
- base.bst
- freedesktop-sdk.bst:components/llvm.bst
- freedesktop-sdk.bst:components/opencl.bst
- freedesktop-sdk.bst:components/patch.bst
- libhwloc.bst
