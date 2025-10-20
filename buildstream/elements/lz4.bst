kind: cmake

sources:
- kind: git_repo
  url: github:lz4/lz4.git
  track: v*
  ref: v1.10.0-0-gebb370ca83af193212df4dcbadcc5d87bc0de2f0

(@): buildstream/elements/base/target.yml

variables:
  command-subdir: build/cmake

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/liblz4.%{dylib-extension}'

  cpe:
    vendor: lz4_project

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/meson.bst
