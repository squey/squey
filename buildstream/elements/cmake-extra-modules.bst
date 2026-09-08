kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git
  url: kde:extra-cmake-modules.git
  ref: 42f0170593df4920068946494c6978ca7005217b

(@): buildstream/elements/base/target.yml

build-depends:
- sdk.bst
- qt6_host.bst
