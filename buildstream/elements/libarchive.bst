kind: cmake
description: Multi-format archive and compression library

sources:
- kind: git_repo
  url: github:libarchive/libarchive.git
  track: v*
  ref: v3.7.7-0-gb439d586f53911c84be5e380445a8a259e19114c

(@): buildstream/elements/base/target.bst

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/libarchive.%{dylib-extension}'

depends:
- sdk.bst