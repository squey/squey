kind: cmake
description: |

    Snappy is a compression/decompression library.

sources:
- kind: tar
  url: github:google/snappy/archive/refs/tags/1.2.2.tar.gz
  ref: 90f74bc1fbf78a6c56b3c4a082a05103b3a56bb17bca1a27e052ea11723292dc

(@): buildstream/elements/base/target.yml

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
