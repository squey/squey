kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: https://anongit.kde.org/kitemmodels.git
  ref: b55c6d90f393d29ed5830e3e39880c732f4c5ea6

build-depends:
- base.bst
- cmake-extra-modules.bst
