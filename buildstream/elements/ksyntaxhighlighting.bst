kind: cmake
description: |

    Syntax highlighting engine for Kate syntax definitions

sources:
- kind: tar

  url: https://github.com/KDE/syntax-highlighting/archive/refs/tags/v5.97.0.tar.gz 
  ref: 2476a85ab680c5a970cbaf8b9af9f1c82f3d2e45315193110bda2dfda3d53ba4

variables:
  cmake-local: |
    -DQT_MAJOR_VERSION=6

build-depends:
- base.bst
- cmake-extra-modules.bst
