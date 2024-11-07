kind: cmake

(@):
  - buildstream/elements/base/target.bst
  - buildstream/elements/qt6.yml

environment:
    CXXFLAGS: "-fuse-ld=%{toolchain-dir}/%{host}-ld"

build-depends:
  (>):
  - sdk.bst
  - qt6_host.bst

depends:
  (>):
  - mysql_target.bst
  - libpq_target.bst
  - brotli.bst
  - icu.bst
  - ncurses.bst
  - openssl.bst
  - zstd.bst