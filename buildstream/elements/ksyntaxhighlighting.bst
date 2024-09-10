kind: cmake
description: |

    Syntax highlighting engine for Kate syntax definitions

sources:
- kind: tar
  url: github:KDE/syntax-highlighting/archive/refs/tags/v6.5.0.tar.gz
  ref: 2c754abe5c226a627f6064096b7d187552e4055a593d689f3fc68a3b5903e2fd

variables:
  cmake-local: |
    -DQT_MAJOR_VERSION=6

build-depends:
- cmake-extra-modules.bst
