kind: cmake
description: Multi-format archive and compression library

sources:
- kind: git_repo
  url: github:libarchive/libarchive.git
  track: v*
  ref: v3.8.9-0-g27cbc7827172698143e440801fc0ba39ccb4f1f5

(@): buildstream/elements/base/target.yml

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/libarchive.%{dylib-extension}'

depends:
- sdk.bst

(?):
- (target_triple != "x86_64-linux-gnu"):
    depends:
      (>):
      - zlib.bst
      - bzip2.bst
      - lz4.bst
      - zstd.bst
