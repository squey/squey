kind: cmake
description: |

  Abseil is an open-source collection of C++ code (compliant to C++14) designed to augment the C++ standard library.

sources:
- kind: git_repo
  url: github:abseil/abseil-cpp.git
  track: '[0-9]*.[0-9]'
  ref: 20260817.0-0-g2065f4ded0558c6f89fee67c8e5228feb4eb960e
- kind: patch
  path: buildstream/patches/abseil-cpp/mingw-no-o-nonblock.patch

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_SKIP_INSTALL_RPATH=ON

config:
  configure-commands:
  # Drop the -lrt of the 'base' target: librt is glibc-only, and the line moves
  # with every abseil release, so it is matched rather than addressed by number.
  - sed '/^ *\$<\$<BOOL:\${LIBRT}>:-lrt>$/d' -i absl/base/CMakeLists.txt
  - "%{cmake}"

depends:
- sdk.bst
