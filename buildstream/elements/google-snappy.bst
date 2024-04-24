kind: cmake
description: |

    Snappy is a compression/decompression library.
    
sources:
- kind: tar
  url: github:google/snappy/archive/refs/tags/1.1.10.tar.gz
  ref: 49d831bffcc5f3d01482340fe5af59852ca2fe76c3e05df0e67203ebbe0f1d90

variables:
  cmake-local: |
    -DSNAPPY_BUILD_TESTS=OFF \
    -DSNAPPY_BUILD_BENCHMARKS=OFF \

depends:
- base.bst


