kind: cmake
description: |

    C language headers for the OpenCL API.

sources:
- kind: tar

  url: https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2023.04.17.tar.gz
  ref: 0ce992f4167f958f68a37918dec6325be18f848dee29a4521c633aae3304915d

- kind: patch
  path: buildstream/patches/opencl-headers/cl.hpp.patch
  strip-level: 1

depends:
- base.bst

