kind: cmake
description: |

    Intel® Threading Building Blocks (Intel® TBB) is a widely used C++ library for shared memory
    parallel programming and heterogeneous computing (intra-node distributed memory programming).

sources:
- kind: tar
  url: github:oneapi-src/oneTBB/archive/v2022.2.0.tar.gz
  ref: f0f78001c8c8edb4bddc3d4c5ee7428d56ae313254158ad1eec49eced57f6a5b

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DTBB_TEST=0

(?):
- (target_triple == "x86_64-w64-mingw32"):
    config:
      configure-commands:
        (<):
        - |
          sed -e 's/:-fstack-clash-protection>/:>/g' -e 's/ -Wshadow / /g' -i cmake/compilers/GNU.cmake
          sed "s/&& !__MINGW32__//g" -i include/oneapi/tbb/profiling.h

depends:
- sdk.bst
