kind: cmake

(@): buildstream/elements/mariadb-connector-c.yml

depends:
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/curl.bst

build-depends:
- base.bst
