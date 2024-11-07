kind: cmake
description: |

    Intel® Threading Building Blocks (Intel® TBB) is a widely used C++ library for shared memory
    parallel programming and heterogeneous computing (intra-node distributed memory programming).

sources:
- kind: tar  
  url: github:oneapi-src/oneTBB/archive/v2022.0.0.tar.gz
  ref: e8e89c9c345415b17b30a2db3095ba9d47647611662073f7fbf54ad48b7f3c2a

(@): buildstream/elements/base/target.bst
  
depends:
- sdk.bst
