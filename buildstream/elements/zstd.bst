kind: cmake

sources:
- kind: git_repo
  url: github:facebook/zstd.git
  track: v*.*.*
  exclude:
  - "*-kernel"
  ref: v1.5.6-0-g794ea1b0afca0f020f4e57b6732332231fb23c70

(@): buildstream/elements/base/target.bst

variables:
  conf-root: "build/cmake"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/libzstd.%{dylib-extension}'

build-depends:
- sdk.bst
- lz4.bst