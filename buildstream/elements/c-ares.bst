kind: autotools
description: |

    c-ares is a C library for asynchronous DNS requests (including name resolves)

sources:
- kind: tar
  url: github:c-ares/c-ares/archive/refs/tags/v1.33.1.tar.gz
  ref: b3d127d8357863eb465053ce9308b79d9b00314f92ec09df056221a1a45c2fef

(@): buildstream/elements/base/target.bst

variables:
  conf-local: |
    --enable-glibtest=no

depends:
- sdk.bst
