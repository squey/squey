kind: cmake
description: |

    C++ wrappers for SIMD intrinsics

sources:
- kind: tar
  url: github:xtensor-stack/xsimd/archive/refs/tags/14.3.0.tar.gz
  ref: b3d50e7a73fbf4642ceef30131c93414901d69eee41c2a5302db650b03e2c792

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
