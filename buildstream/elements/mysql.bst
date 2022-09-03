kind: cmake
description: |

    The world's most popular open source database 

sources:
- kind: tar

  url: https://github.com/mysql/mysql-server/archive/refs/tags/mysql-8.0.30.tar.gz
  ref: e76636197f9cb764940ad8d800644841771def046ce6ae75c346181d5cdd879a

  
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
