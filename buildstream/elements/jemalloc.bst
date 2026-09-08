kind: autotools
description: |

    jemalloc is a general purpose malloc(3) implementation that emphasizes
    fragmentation avoidance and scalable concurrency support.

sources:
- kind: tar
  url: github:jemalloc/jemalloc/archive/refs/tags/5.3.1.tar.gz
  ref: 7b30f6116f11d736badd48c903cba2b3344a88d62e3a7b892434f870e860b1c0
- kind: patch
  path: buildstream/patches/jemalloc/gcc16-throw-bad-alloc.patch

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
