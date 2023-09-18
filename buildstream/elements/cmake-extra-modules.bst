kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git

  url: kde:extra-cmake-modules.git
  ref: 553e9e8657b724139a96a68cdbefc6c261b662cb

depends:
- base.bst
- qt6.bst
