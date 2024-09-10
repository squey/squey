kind: autotools
description: |

    The fastest malloc we’ve seen; works particularly well with threads
    and STL. Also: thread-friendly heap-checker, heap-profiler, and
    cpu-profiler.

sources:
- kind: tar

  url: github:gperftools/gperftools/archive/refs/tags/gperftools-2.15.tar.gz
  ref: 3918ff2e21bb3dbb5a801e1daf55fb20421906f7c42fbb482bede7bdc15dfd2e

config:

  # Override autotools element default configure-commands
  configure-commands:
  - |
    ./autogen.sh
    %{configure}

depends:
- base.bst

build-depends:
- freedesktop-sdk.bst:components/patch.bst
