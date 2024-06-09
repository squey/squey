kind: cmake
description: |

    mold is a faster drop-in replacement for existing Unix linkers.

sources:
- kind: tar

  url: github:rui314/mold/archive/refs/tags/v2.32.0.tar.gz
  ref: 4b7e4146ea0f52be9adae8b417399f3676a041e65b55e3f25f088120d30a320b

config:
  install-commands:
  - |
    %{make-install}
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.bfd"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.gold"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld"

depends:
- base.bst


