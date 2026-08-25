kind: cmake
description: |

    Snappy is a compression/decompression library.

sources:
- kind: tar
  url: github:google/snappy/archive/refs/tags/1.2.2.tar.gz
  ref: 90f74bc1fbf78a6c56b3c4a082a05103b3a56bb17bca1a27e052ea11723292dc

(@): buildstream/elements/base/target.yml

# SNAPPY_REQUIRE_AVX2 adds -mavx -mbmi2, which lets snappy.cc take its __BMI2__
# path : the bzhi and shrx it emits in DecompressAllTags and FindMatchLength are
# Haswell instructions, so every Parquet file read with the default codec died with
# SIGILL on the Sandy Bridge and Ivy Bridge machines the crash reports came from.
# Keeping them off leaves snappy on the x86-64-v2 baseline the rest of the build uses.
variables:
  cmake-local: >-
    -DSNAPPY_BUILD_TESTS=OFF
    -DBUILD_SHARED_LIBS=ON
    -DSNAPPY_BUILD_BENCHMARKS=OFF

depends:
- sdk.bst
