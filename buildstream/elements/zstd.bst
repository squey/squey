kind: cmake

sources:
- kind: tar
  url: github:facebook/zstd/archive/refs/tags/v1.5.7.tar.gz
  ref: 37d7284556b20954e56e1ca85b80226768902e2edabd3b649e9e72c0c9012ee3

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