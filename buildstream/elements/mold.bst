kind: cmake
description: |

    mold is a faster drop-in replacement for existing Unix linkers.

sources:
- kind: tar

  url: github:rui314/mold/archive/refs/tags/v2.2.0.tar.gz
  ref: 78ddddaaa004e50f8d92a13d8e792a46a1b37745fab48d39ad16aeb5a776e7c6

config:
  install-commands:
  - |
    %{make-install}
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.bfd"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.gold"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld"

depends:
- base.bst


