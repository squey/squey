kind: autotools
description: |

    The fastest malloc we’ve seen; works particularly well with threads
    and STL. Also: thread-friendly heap-checker, heap-profiler, and
    cpu-profiler.

sources:
- kind: tar
  url: github:gperftools/gperftools/archive/refs/tags/gperftools-2.16.tar.gz
  ref: 737be182b4e42f5c7f595da2a7aa59ce0489a73d336d0d16847f2aa52d5221b4

(@): buildstream/elements/base/target.bst

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
