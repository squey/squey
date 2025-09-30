kind: cmake

(@):  buildstream/elements/clang.yml

variables:
  cmake-local-extra: >-
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DLLVM_DEFAULT_TARGET_TRIPLE=%{build-triplet}

depends:
  (>):
  - base.bst
