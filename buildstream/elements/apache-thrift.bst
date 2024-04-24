kind: cmake
description: |

    Apache Thrift is a lightweight, language-independent software stack for point-to-point RPC implementation.
    
sources:
- kind: tar
  url: github:apache/thrift/archive/refs/tags/v0.19.0.tar.gz
  ref: 6428911db505702c51f7d993155a4a4c8afee83fdd021b52f2eccd8d34780629

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