kind: cmake
description: |

    PoCL is a portable open source (MIT-licensed) implementation of the OpenCL standard.

sources:
- kind: tar
  url: github:pocl/pocl/archive/refs/tags/v6.0.tar.gz
  ref: de9710223fc1855f833dbbf42ea2681e06aa8ec0464f0201104dc80a74dfd1f2

- kind: patch
  path: buildstream/patches/portablecl/FindTBB.cmake.patch
  strip-level: 1

variables:
  cmake-local: |
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DINSTALL_OPENCL_HEADERS=1 \
    -DEXTRA_HOST_LD_FLAGS=-L%{prefix}/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DKERNELLIB_HOST_CPU_VARIANTS=distro \
    -DENABLE_TBB_DEVICE=1

build-depends:
- freedesktop-sdk.bst:components/llvm.bst

depends:
- base.bst
- freedesktop-sdk.bst:components/opencl.bst
- freedesktop-sdk.bst:components/patch.bst
- freedesktop-sdk.bst:components/python3.bst
- tbb.bst
- libhwloc.bst
- clang.bst
- binutils.bst
