kind: autotools
description: |

    c-ares is a C library for asynchronous DNS requests (including name resolves)

sources:
- kind: tar
  url: github:c-ares/c-ares/archive/refs/tags/v1.34.8.tar.gz
  ref: ddf8be8b179ef6ff656e7413da93cf0f3038c627aca29ce9aae30f66e5b32bc1

(@): buildstream/elements/base/target.yml

variables:
  conf-local: |
    --enable-glibtest=no

depends:
- sdk.bst
