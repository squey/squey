kind: cmake
description: |

    C language headers for the OpenCL API.

sources:
- kind: tar
  url: github:KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.10.24.tar.gz
  ref: 159f2a550592bae49859fee83d372acd152328fdf95c0dcd8b9409f8fad5db93

- kind: patch
  path: buildstream/patches/opencl-headers/cl.hpp.patch
  strip-level: 1

(@): buildstream/elements/base/target.bst

variables:
  make-args: ""
  cmake-global: ""

build-depends:
- sdk.bst

