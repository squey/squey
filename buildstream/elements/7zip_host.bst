kind: manual
description: |

    7-Zip is a file archiver with a high compression ratio.

sources:
- kind: remote
  url: github:ip7z/7zip/releases/download/26.03/7z2603-linux-x86.tar.xz
  ref: 1ac4d68723f457ce000a069f5453051aa5381765ce5c77b1d5a02ea8cfba9af7

config:
  install-commands:
  - |
    tar Jxvf *.tar.xz
    install -D 7zzs -t %{install-root}/%{bindir}

build-depends:
- base.bst
