kind: cmake
description: |

    mold is a faster drop-in replacement for existing Unix linkers.

sources:
- kind: tar
  url: github:rui314/mold/archive/refs/tags/v2.40.4.tar.gz
  ref: 69414c702ec1084e1fa8ca16da24f167f549e5e11e9ecd5d70a8dcda6f08c249

config:
  install-commands:
  - |
    %{make-install}
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.bfd"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.gold"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld"

depends:
- base.bst
