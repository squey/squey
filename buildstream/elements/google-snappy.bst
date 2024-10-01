kind: cmake
description: |

    Snappy is a compression/decompression library.
    
sources:
- kind: tar
  url: github:google/snappy/archive/refs/tags/1.2.1.tar.gz
  ref: 736aeb64d86566d2236ddffa2865ee5d7a82d26c9016b36218fcc27ea4f09f86

variables:
  cmake-local: |
    -DSNAPPY_BUILD_TESTS=OFF \
    -DSNAPPY_BUILD_BENCHMARKS=OFF \

depends:
- base.bst


