kind: cmake
description: |

    mold is a faster drop-in replacement for existing Unix linkers.

sources:
- kind: tar

  url: github:rui314/mold/archive/refs/tags/v2.33.0.tar.gz
  ref: 37b3aacbd9b6accf581b92ba1a98ca418672ae330b78fe56ae542c2dcb10a155

config:
  install-commands:
  - |
    %{make-install}
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.bfd"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld.gold"
    ln -s "%{prefix}/bin/mold" "%{install-root}%{prefix}/bin/ld"

depends:
- base.bst


