kind: cmake

sources:
- kind: git_repo
  url: github:google/brotli.git
  track: v*
  exclude:
  - "*rc*"
  ref: v1.2.0-0-g028fb5a23661f123017c060daa546b55cf4bde29

(@): buildstream/elements/base/target.yml

config:
  configure-commands:
    (<):
    - sed '116d' -i CMakeLists.txt

public:
  bst:
    split-rules:
      devel:
        (>):
        - "%{bindir}/brotli"
        - "%{libdir}/libbrotlicommon.%{dylib-extension}"
        - "%{libdir}/libbrotlidec.%{dylib-extension}"
        - "%{libdir}/libbrotlienc.%{dylib-extension}"

build-depends:
- sdk.bst
