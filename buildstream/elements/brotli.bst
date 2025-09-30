kind: cmake

sources:
- kind: git_repo
  url: github:google/brotli.git
  track: v*
  exclude:
  - "*rc*"
  ref: v1.1.0-0-ged738e842d2fbdf2d6459e39267a633c4a9b2f5d

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
