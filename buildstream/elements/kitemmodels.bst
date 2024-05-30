kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: kde:kitemmodels.git
  ref: 18441664cc52050477b16a06b87bfebb4b3fc5c0

variables:
  cmake-local: |
    -DQT_MAJOR_VERSION=6 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fpermissive"

depends:
- base.bst

build-depends:
- cmake-extra-modules.bst
