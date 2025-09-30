kind: cmake
description: |

    C language headers for the OpenCL API.

sources:
- kind: tar
  url: github:KhronosGroup/OpenCL-Headers/archive/refs/tags/v2025.07.22.tar.gz
  ref: 98f0a3ea26b4aec051e533cb1750db2998ab8e82eda97269ed6efe66ec94a240

- kind: patch
  path: buildstream/patches/opencl-headers/cl.hpp.patch
  strip-level: 1


(@): buildstream/elements/base/target.yml

variables:
  make-args: ""
  cmake-global: ""

build-depends:
- sdk.bst
