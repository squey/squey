kind: cmake
description: |

    FUSE module for .dmg files (containing an HFS+ filesystem) 

sources:
- kind: git
  url: github:darlinghq/darling-dmg.git
  ref: a36bf0c07b16675b446377890c5f6f74563f84dd

variables:
    cmake-local: >-
        -DCMAKE_CXX_STANDARD=17
        -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="%{includedir}"
        -DCMAKE_EXE_LINKER_FLAGS="-L%{libdir}"

depends:
- base.bst
- libfuse.bst
- freedesktop-sdk.bst:components/libxml2.bst
- freedesktop-sdk.bst:components/icu.bst