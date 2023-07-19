kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: https://anongit.kde.org/kitemmodels.git
  ref: 064c9699a8ea82b51b57c21e4ae2d02c783a8056

variables:
  cmake-local: |
    -DQT_MAJOR_VERSION=6 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fpermissive"

depends:
- base.bst

build-depends:
- cmake-extra-modules.bst
