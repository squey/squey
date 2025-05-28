kind: cmake

sources:
- kind: tar
  url: github:mariadb-corporation/mariadb-connector-c/archive/refs/tags/v3.4.5.tar.gz
  ref: 07803adff502edf9b294ba1953cd99e2729d728bcb13c20f823633f7507040a6

(@): buildstream/elements/mariadb-connector-c.yml

depends:
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/curl.bst

build-depends:
- base.bst
