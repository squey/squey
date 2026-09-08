kind: cmake
description: |

    The OpenCL ICD Loader project from Khronos Group

sources:
- kind: tar
  url: github:KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2026.05.29.tar.gz
  ref: 48fd0c5181db7cd046f4f731d5955694892e10998d49d09ee0d997e7e04fd939

(@): buildstream/elements/base/target.yml

build-depends:
- opencl-headers.bst
- sdk.bst
