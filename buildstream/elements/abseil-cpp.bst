kind: cmake
description: |

  Abseil is an open-source collection of C++ code (compliant to C++14) designed to augment the C++ standard library.

sources:
- kind: git_repo
  url: github:abseil/abseil-cpp.git
  track: '[0-9]*.[0-9]'
  ref: 20240722.0-0-g4447c7562e3bc702ade25105912dce503f0c4010

(@): buildstream/elements/base/target.bst

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_SKIP_INSTALL_RPATH=ON

config:
  configure-commands:
  - sed '243d' -i absl/base/CMakeLists.txt
  - "%{cmake}"

depends:
- sdk.bst