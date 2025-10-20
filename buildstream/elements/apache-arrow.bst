kind: cmake
description: |

    Apache Arrow is a development platform for in-memory analytics. It contains a set of technologies that enable big data systems to process and move data fast.

sources:
- kind: tar
  url: github:apache/arrow/releases/download/apache-arrow-21.0.0/apache-arrow-21.0.0.tar.gz
  ref: 5d3f8db7e72fb9f65f4785b7a1634522e8d8e9657a445af53d4a34a3849857b5

(@): buildstream/elements/base/target.yml

variables:
  source-dir: "."
  cmake-local: >-
    -DCMAKE_BUILD_TYPE=Release
    -DARROW_PARQUET=ON
    -DPARQUET_REQUIRE_ENCRYPTION=ON
    -DThrift_SOURCE=SYSTEM
    -Djemalloc_SOURCE=SYSTEM
    -Dutf8proc_SOURCE=SYSTEM
    -DARROW_WITH_SNAPPY=ON
    -DARROW_WITH_BROTLI=ON
    -DARROW_WITH_BZ2=ON
    -DARROW_WITH_ZSTD=ON
    -DARROW_WITH_ZLIB=ON
    -DARROW_WITH_LZ4=ON
    -DARROW_JEMALLOC=OFF
    -DARROW_MIMALLOC=OFF
    -DARROW_COMPUTE=ON
    -DARROW_JSON=ON

config:
  configure-commands:
    - |
      cd cpp
      if [ "%{target-platform}" = "windows" ] ; then
        find %{source-dir} -name io_util.cc -exec sed 's/Windows.h/windows.h/' -i {} \;
      fi
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
  - (target_triple != "x86_64-linux-gnu"):
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
