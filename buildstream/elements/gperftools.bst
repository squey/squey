kind: autotools
description: |

    The fastest malloc we’ve seen; works particularly well with threads
    and STL. Also: thread-friendly heap-checker, heap-profiler, and
    cpu-profiler.

sources:
- kind: tar
  url: github:gperftools/gperftools/archive/refs/tags/gperftools-2.18.1.tar.gz
  ref: 172d27e6f6c1fa69df4be30bc61ea35ed225b74cd3a15500e2d75e981478fb2b

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
