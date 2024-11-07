kind: cmake
description: |

    Snappy is a compression/decompression library.
    
sources:
- kind: tar
  url: github:google/snappy/archive/refs/tags/1.2.1.tar.gz
  ref: 736aeb64d86566d2236ddffa2865ee5d7a82d26c9016b36218fcc27ea4f09f86

(@): buildstream/elements/base/target.bst

(?):
  - (target_triple == "x86_64-linux-gnu" or target_triple == "x86_64-apple-darwin"):
      variables:
        cmake-local-extra: >-
          -DSNAPPY_REQUIRE_AVX=ON
          -DSNAPPY_REQUIRE_AVX2=ON

variables:
  cmake-local: >-
    -DSNAPPY_BUILD_TESTS=OFF
    -DBUILD_SHARED_LIBS=ON
    -DSNAPPY_BUILD_BENCHMARKS=OFF
    %{cmake-local-extra}

depends:
- sdk.bst


