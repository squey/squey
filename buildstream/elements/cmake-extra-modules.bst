kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: kde:extra-cmake-modules.git
  ref: 54646d3b5dd8d1112d8be370bd34a3fc63ab067a

depends:
- base.bst
- qt6.bst
