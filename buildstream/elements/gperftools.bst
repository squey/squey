kind: autotools
description: |

    The fastest malloc we’ve seen; works particularly well with threads
    and STL. Also: thread-friendly heap-checker, heap-profiler, and
    cpu-profiler.

sources:
- kind: tar
  url: github:gperftools/gperftools/archive/refs/tags/gperftools-2.17.2.tar.gz
  ref: 885dbbf1f25a922de0cdc78b0703c3ab93c43850e1d2f7c889e41be7c824c53d

(@): buildstream/elements/base/target.yml

config:

  # Override autotools element default configure-commands
  configure-commands:
  - |
    ./autogen.sh
    %{configure}

depends:
- sdk.bst

build-depends:
- freedesktop-sdk.bst:components/patch.bst
