kind: cmake
description: |

    C language headers for the OpenCL API.

sources:
- kind: tar
  url: github:KhronosGroup/OpenCL-Headers/archive/refs/tags/v2026.05.29.tar.gz
  ref: d9e6c48357de5002da11ce45de600e0c3ffe6ab4f628a3b9fe2b38603161658a

- kind: patch
  path: buildstream/patches/opencl-headers/cl.hpp.patch
  strip-level: 1


(@): buildstream/elements/base/target.yml

variables:
  make-args: ""
  cmake-global: ""

build-depends:
- sdk.bst
