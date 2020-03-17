kind: cmake
description: |

    The world's most popular open source database 

sources:
- kind: tar

  url: https://github.com/mysql/mysql-server/archive/mysql-5.7.29.tar.gz
  ref: 991aad054fcf3ccf48e823478122bdf34c69e892cafd7fb80ec017dcc7114d0c

  
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
