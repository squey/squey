kind: autotools
description: |

    The fastest malloc we’ve seen; works particularly well with threads
    and STL. Also: thread-friendly heap-checker, heap-profiler, and
    cpu-profiler.

sources:
- kind: tar

  url: https://github.com/gperftools/gperftools/archive/gperftools-2.6.3.tar.gz
  ref: 0c067fbef016daea0ff5c73135444624bf6efa89662c1537e54dad7a2b037c1b

config:

  # Override autotools element default configure-commands
  configure-commands:
  - |
    ./autogen.sh
    %{configure}

depends:
- base.bst
