kind: meson

sources:
- kind: tar
  url: github:simd-everywhere/simde/archive/refs/tags/v0.8.2.tar.gz
  ref: ed2a3268658f2f2a9b5367628a85ccd4cf9516460ed8604eed369653d49b25fb

(@): buildstream/elements/base/target.bst

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/meson.bst

