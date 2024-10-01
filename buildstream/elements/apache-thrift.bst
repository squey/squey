kind: cmake
description: |

    Apache Thrift is a lightweight, language-independent software stack for point-to-point RPC implementation.
    
sources:
- kind: tar
  url: github:apache/thrift/archive/refs/tags/v0.20.0.tar.gz
  ref: cd7b829d3d9d87f9f7d708e004eef7629789591ee1d416f4741913bc33e5c27d

variables:
  cmake-local: |
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_COMPILER=OFF \
    -DWITH_PYTHON=OFF \
    -DWITH_JAVA=OFF \
    -DWITH_JAVASCRIPT=OFF \
    -DWITH_NODEJS=OFF \
    -DWITH_KOTLIN=OFF

config:
  configure-commands:
    - |
      %{cmake}

depends:
- base.bst

build-depends:
- boost.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst