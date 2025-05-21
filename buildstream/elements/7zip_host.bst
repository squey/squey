kind: manual
description: |

    7-Zip is a file archiver with a high compression ratio.

sources:
- kind: remote
  url: github:ip7z/7zip/releases/download/24.08/7z2408-linux-x64.tar.xz
  ref: 9b3a2edf0644a2b00b0a39ab6843f426128e4a6f6f98c40f8c02a668e5b5b59b

config:
  install-commands:
  - |
    tar Jxvf *.tar.xz
    install -D 7zz -t %{install-root}/%{bindir}

build-depends:
- base.bst