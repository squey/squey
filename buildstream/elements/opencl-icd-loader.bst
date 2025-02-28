kind: cmake
description: |

    The OpenCL ICD Loader project from Khronos Group

sources:
- kind: tar
  url: github:KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2024.10.24.tar.gz
  ref: 95f2f0cda375b13d2760290df044ebea9c6ff954a7d7faa0867422442c9174dc

(@): buildstream/elements/base/target.bst

build-depends:
- opencl-headers.bst
- sdk.bst

