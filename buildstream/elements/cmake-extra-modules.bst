kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: kde:extra-cmake-modules.git
  ref: 07e04d248b0639fe039f45769188dbd8c835d91c

depends:
- base.bst
- qt6.bst
