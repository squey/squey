kind: cmake
description: |

    PoCL is a portable open source (MIT-licensed) implementation of the OpenCL standard.

sources:
- kind: tar
  url: github:pocl/pocl/archive/refs/tags/v6.0-RC2.tar.gz
  ref: 7b7aea699c73dbfc6afa4013c651380859ba4526f1729809772486b7c35351b9

variables:
  cmake-local: |
    -DINSTALL_OPENCL_HEADERS=1 \
    -DEXTRA_HOST_LD_FLAGS=-L%{prefix}/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLC_HOST_CPU=corei7

depends:
- base.bst
- freedesktop-sdk.bst:components/opencl.bst
- freedesktop-sdk.bst:components/patch.bst
- freedesktop-sdk.bst:components/python3.bst
- libhwloc.bst
- clang.bst
- binutils.bst
