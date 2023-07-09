kind: cmake
description: |

    The Open Standard for Parallel Programming of Heterogeneous Systems

sources:
- kind: tar
  url: http://portablecl.org/downloads/pocl-3.0.tar.gz
  ref: a3fd3889ef7854b90b8e4c7899c5de48b7494bf770e39fba5ad268a5cbcc719d

- kind: patch
  path: buildstream/patches/portablecl/cl.hpp.patch
  strip-level: 1
  
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
- libhwloc.bst
