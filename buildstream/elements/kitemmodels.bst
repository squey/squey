kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: https://anongit.kde.org/kitemmodels.git
  ref: 78cde10a52194c24db7b676b13dbd8e6847c2424

variables:
  cmake-local: |
    -DQT_MAJOR_VERSION=6 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fpermissive"

build-depends:
- base.bst
- cmake-extra-modules.bst
