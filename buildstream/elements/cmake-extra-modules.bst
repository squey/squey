kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git
  url: kde:extra-cmake-modules.git
  ref: 67c974b0e34feee683f4d905dde33a2b6abac2f7

(@): buildstream/elements/base/target.yml

build-depends:
- sdk.bst
- qt6_host.bst
