kind: autotools

(@):
  - buildstream/elements/base/target.bst
  - buildstream/elements/binutils.yml

build-depends:
  (>):
  - sdk.bst

depends:
- zlib.bst
    