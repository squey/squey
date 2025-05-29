kind: cmake

sources:
- kind: tar
  url: github:mariadb-corporation/mariadb-connector-c/archive/refs/tags/v3.4.5.tar.gz
  ref: 07803adff502edf9b294ba1953cd99e2729d728bcb13c20f823633f7507040a6

(@): 
  - buildstream/elements/mariadb-connector-c.yml
  - buildstream/elements/base/target.bst

build-depends:
- sdk.bst

depends:
- openssl.bst
- zlib.bst
- zstd.bst

(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      depends:
        (>):
        - curl.bst
  - target_triple == "aarch64-apple-darwin":
      variables:
        cmake-local-extra: "-DCMAKE_C_FLAGS=-Wno-inline-asm"


