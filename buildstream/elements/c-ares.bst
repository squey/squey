kind: autotools
description: |

    c-ares is a C library for asynchronous DNS requests (including name resolves)

sources:
- kind: tar
  url: github:c-ares/c-ares/archive/refs/tags/v1.34.5.tar.gz
  ref: dcd919635f01b7c8c9c2f5fb38063cd86500f7c6d4d32ecf4deff5e3497fb157

(@): buildstream/elements/base/target.yml

variables:
  conf-local: |
    --enable-glibtest=no

depends:
- sdk.bst
