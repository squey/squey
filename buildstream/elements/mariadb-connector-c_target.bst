kind: cmake

(@):
  - buildstream/elements/mariadb-connector-c.yml
  - buildstream/elements/base/target.yml

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
        cmake-local-extra: "-DCMAKE_C_FLAGS='-Wno-inline-asm -Wno-error=default-const-init-var-unsafe'"
  - target_triple == "x86_64-apple-darwin":
      variables:
        cmake-local-extra: "-DCMAKE_C_FLAGS=-Wno-error=default-const-init-var-unsafe"
