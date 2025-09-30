kind: cmake
description: |

  Abseil is an open-source collection of C++ code (compliant to C++14) designed to augment the C++ standard library.

sources:
- kind: git_repo
  url: github:abseil/abseil-cpp.git
  track: '[0-9]*.[0-9]'
  ref: 20250814.1-0-gd38452e1ee03523a208362186fd42248ff2609f6

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_SKIP_INSTALL_RPATH=ON

config:
  configure-commands:
  - sed '250d' -i absl/base/CMakeLists.txt
  - "%{cmake}"

depends:
- sdk.bst
