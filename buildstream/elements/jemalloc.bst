kind: autotools
description: |

    jemalloc is a general purpose malloc(3) implementation that emphasizes
    fragmentation avoidance and scalable concurrency support. 
    
sources:
- kind: tar
  url: github:jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
  ref: 2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa

depends:
- base.bst
