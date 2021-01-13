kind: cmake
description: |

    Syntax highlighting engine for Kate syntax definitions

sources:
- kind: tar

  url: https://github.com/KDE/syntax-highlighting/archive/v5.76.0.tar.gz 
  ref: 67b5116b20709ebdf2684a6684a77311f9274d17797ee7fb9dd43c2ff83cb390

build-depends:
- base.bst
- cmake-extra-modules.bst
