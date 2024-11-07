kind: cmake
description: |

    Apache Thrift is a lightweight, language-independent software stack for point-to-point RPC implementation.
    
sources:
- kind: tar
  url: github:apache/thrift/archive/refs/tags/v0.20.0.tar.gz
  ref: cd7b829d3d9d87f9f7d708e004eef7629789591ee1d416f4741913bc33e5c27d

(@): buildstream/elements/base/target.bst

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    -DBUILD_COMPILER=OFF
    -DWITH_PYTHON=OFF
    -DWITH_JAVA=OFF
    -DWITH_JAVASCRIPT=OFF
    -DWITH_NODEJS=OFF
    -DWITH_KOTLIN=OFF

config:
  configure-commands:
    - |
      %{cmake}

build-depends:
- sdk.bst

(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      depends:
        - boost.bst
        - openssl.bst
        - zlib.bst
        - zstd.bst
        - lz4.bst
  - target_triple == "x86_64-linux-gnu":
      depends:
        - boost.bst
