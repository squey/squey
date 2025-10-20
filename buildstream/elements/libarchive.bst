kind: cmake
description: Multi-format archive and compression library

sources:
- kind: git_repo
  url: github:libarchive/libarchive.git
  track: v*
  ref: v3.8.1-0-g9525f90ca4bd14c7b335e2f8c84a4607b0af6bdf

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
