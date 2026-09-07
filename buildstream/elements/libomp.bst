kind: cmake
description: |

    LLVM OpenMP library implementation

# The per-component openmp-*.src.tar.xz and cmake-*.src.tar.xz tarballs are no
# longer published from LLVM 22 on, so the monorepo archive is used instead. It
# is the very one clang.yml builds from, so the two share a single download.
sources:
- kind: tar
  url: github:llvm/llvm-project/archive/refs/tags/llvmorg-22.1.8.tar.gz
  ref: ad18b70e287954c3d62bc7e0b86e7b7af2adf87bcfce21c15fe717f101d7aace

(@): buildstream/elements/base/target.yml

variables:
  # openmp/CMakeLists.txt reaches for ../cmake/Modules, which the monorepo lays
  # out the way it expects.
  conf-root: "openmp"
  cmake-local: >-
    -DCMAKE_BUILD_TYPE=Release

depends:
- sdk.bst
- freedesktop-sdk.bst:components/llvm.bst
- freedesktop-sdk.bst:components/python3.bst
