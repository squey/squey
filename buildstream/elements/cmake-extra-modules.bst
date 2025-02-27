kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git
  url: kde:extra-cmake-modules.git
  ref: 7dd28cc56c339c3f8fb356f7c53c0e8f61433d81

(@): buildstream/elements/base/target.bst

build-depends:
- sdk.bst
- qt6_host.bst
