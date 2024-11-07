kind: cmake
description: |

    Apache Arrow is a development platform for in-memory analytics. It contains a set of technologies that enable big data systems to process and move data fast.
    
sources:
- kind: tar
  url: github:apache/arrow/archive/refs/tags/apache-arrow-18.0.0.tar.gz
  ref: 9c473f2c9914c59ab571761c9497cf0e5cfd3ea335f7782ccc6121f5cb99ae9b

(@): buildstream/elements/base/target.bst

variables:
  cmake-local: >-
    -DCMAKE_BUILD_TYPE=Release
    -DARROW_PARQUET=ON
    -DPARQUET_REQUIRE_ENCRYPTION=ON
    -Djemalloc_SOURCE=SYSTEM
    -Dutf8proc_SOURCE=SYSTEM
    -Dutf8proc_LIB=%{prefix}/lib/libutf8proc.a
    -DARROW_WITH_SNAPPY=ON
    -DARROW_WITH_BROTLI=ON
    -DARROW_WITH_BZ2=ON
    -DARROW_WITH_ZSTD=ON
    -DARROW_WITH_ZLIB=ON
    -DARROW_WITH_LZ4=ON
    -DARROW_JEMALLOC=OFF
    -DARROW_COMPUTE=ON

config:
  configure-commands:
    - |
      cd cpp
      %{cmake}

  build-commands:
    - |
      cd cpp
      %{make}

  install-commands:
    - |
      cd cpp
      %{make-install}

build-depends:
- sdk.bst

depends:
- google-snappy.bst
- apache-thrift.bst
- jemalloc.bst
- xsimd.bst
- boost.bst
- rapidjson.bst
- google-re2.bst
- utf8proc.bst

(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      depends:
        (>):
        - bzip2.bst
        - lz4.bst
        - zstd.bst
        - brotli.bst
        - zlib.bst
        - openssl.bst
  - target_triple == "x86_64-linux-gnu":
      depends:
        (>):
        - freedesktop-sdk.bst:bootstrap/bzip2.bst
        - freedesktop-sdk.bst:bootstrap/zlib.bst
        - freedesktop-sdk.bst:components/brotli.bst
        - freedesktop-sdk.bst:components/openssl.bst
        - freedesktop-sdk.bst:components/lz4.bst
        - freedesktop-sdk.bst:components/zstd.bst
