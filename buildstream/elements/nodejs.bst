kind: autotools
description: |

    As an asynchronous event-driven JavaScript runtime, Node.js is designed to build scalable network applications.

sources:
- kind: tar
  url: github:nodejs/node/archive/refs/tags/v22.8.0.tar.gz
  ref: cce84db6209366d35a9bffddd25294d97a3432c04f14e06ade7617d585073c7f

config:
  configure-commands:
  - ./configure --prefix=%{prefix} --ninja

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst