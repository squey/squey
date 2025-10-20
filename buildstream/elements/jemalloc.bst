kind: autotools
description: |

    jemalloc is a general purpose malloc(3) implementation that emphasizes
    fragmentation avoidance and scalable concurrency support.

sources:
- kind: tar
  url: github:jemalloc/jemalloc/archive/refs/tags/5.3.0.tar.gz
  ref: ef6f74fd45e95ee4ef7f9e19ebe5b075ca6b7fbe0140612b2a161abafb7ee179

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
