kind: cmake
description: |

    The world's most popular open source database 

sources:
- kind: tar

  url: https://github.com/mysql/mysql-server/archive/refs/tags/mysql-8.0.33.tar.gz
  ref: d986cbcb3028676eef037d94f82eaac79caf4af24e1b4c304dab49896fc5eaf7

  
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
