kind: cmake
description: |

    The OpenCL ICD Loader project from Khronos Group

sources:
- kind: tar
  url: github:KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2025.07.22.tar.gz
  ref: dff7a0b11ad5b63a669358e3476e3dc889a4a361674e5b69b267b944d0794142

(@): buildstream/elements/base/target.yml

build-depends:
- opencl-headers.bst
- sdk.bst
