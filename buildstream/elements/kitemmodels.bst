kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git
  url: kde:kitemmodels.git
  ref: fa926680500bac596b2fce92f82ca19545c3085e

variables:
  cmake-local: |
    -DQT_MAJOR_VERSION=6 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fpermissive"

depends:
- base.bst

build-depends:
- cmake-extra-modules.bst
