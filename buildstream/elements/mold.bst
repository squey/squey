kind: cmake
description: |

    mold is a faster drop-in replacement for existing Unix linkers.

sources:
- kind: tar
  url: github:rui314/mold/archive/refs/tags/v2.42.0.tar.gz
  ref: 6c0f3308c5b3159a369202d970922ad819bab1bfcb5a3b3c06a723d19f65373e

config:
  install-commands:
  - |
    %{make-install}
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.bfd"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.gold"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld"

depends:
- base.bst
