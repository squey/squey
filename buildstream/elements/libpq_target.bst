kind: autotools

(@): 
  - buildstream/elements/libpq.yml
  - buildstream/elements/base/target.bst

environment:
  LDFLAGS: "-L%{libdir}"
  CPPFLAGS: "-I%{includedir}"

depends:
- sdk.bst
- openssl.bst
- icu.bst
- lz4.bst
