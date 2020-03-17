kind: cmake
description: |

    The Open Standard for Parallel Programming of Heterogeneous Systems

sources:
- kind: tar
  url: https://github.com/pocl/pocl/archive/v1.4.tar.gz
  ref: ec237faa83bb1c803fbdf7c6e83d8a2ad68b6f0ed1879c3aa16c0e1dcc478742
  
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
