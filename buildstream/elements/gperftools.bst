kind: autotools
description: |

    The fastest malloc we’ve seen; works particularly well with threads
    and STL. Also: thread-friendly heap-checker, heap-profiler, and
    cpu-profiler.

sources:
- kind: tar

  url: https://github.com/gperftools/gperftools/archive/gperftools-2.8.tar.gz
  ref: b09193adedcc679df2387042324d0d54b93d35d062ea9bff0340f342a709e860

config:

  # Override autotools element default configure-commands
  configure-commands:
  - |
    ./autogen.sh
    %{configure}

depends:
- base.bst
