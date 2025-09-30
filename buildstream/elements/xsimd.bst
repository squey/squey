kind: cmake
description: |

    C++ wrappers for SIMD intrinsics

sources:
- kind: tar
  url: github:xtensor-stack/xsimd/archive/refs/tags/13.2.0.tar.gz
  ref: edd8cd3d548c185adc70321c53c36df41abe64c1fe2c67bc6d93c3ecda82447a

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
