kind: cmake
description: |

    The world's most popular open source database 

sources:
- kind: tar

  url: github:mysql/mysql-server/archive/refs/tags/mysql-9.0.1.tar.gz
  ref: 54a8a99a810c2c8ca51f11c7e5a764018066db1b75ba92c509c3794bd0cd552c

  
variables:
  cmake-local: |
    -DCMAKE_CXX_COMPILER="g++" \
    -DCMAKE_C_COMPILER="gcc" \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -L%{prefix}/lib" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITHOUT_SERVER=ON

depends:
- base.bst
- boost.bst
- freedesktop-sdk.bst:components/openssl.bst
