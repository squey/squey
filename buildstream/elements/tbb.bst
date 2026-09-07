kind: cmake
description: |

    Intel® Threading Building Blocks (Intel® TBB) is a widely used C++ library for shared memory
    parallel programming and heterogeneous computing (intra-node distributed memory programming).

sources:
- kind: tar
  url: github:oneapi-src/oneTBB/archive/v2023.1.0.tar.gz
  ref: 191288b52e1e6b17198000b64d77d194bb65e791be46ebc606e9b091781e2070

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
