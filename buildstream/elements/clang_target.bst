kind: cmake

(@):
  - buildstream/elements/base/target.bst
  - buildstream/elements/clang.yml

variables:
  cmake-local-extra:
    -DLLVM_DEFAULT_TARGET_TRIPLE=%{target_triple}

build-depends:
  (>):
  - sdk.bst
  - binutils.bst