kind: cmake
description: |

    The Open Standard for Parallel Programming of Heterogeneous Systems

sources:
- kind: tar
  url: http://portablecl.org/downloads/pocl-1.6.tar.gz
  ref: b0a4c0c056371b6f0db726b88fbb76bbee94948fb2abd4dbc8d958f7c42f766c
  
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
