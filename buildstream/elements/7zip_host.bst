kind: manual
description: |

    7-Zip is a file archiver with a high compression ratio.

sources:
- kind: remote
  url: github:ip7z/7zip/releases/download/25.01/7z2501-linux-x86.tar.xz
  ref: 62a47626eb9ce2886de3fec6f201f6a4210f14ba61b30a16141b953aed705207

config:
  install-commands:
  - |
    tar Jxvf *.tar.xz
    install -D 7zzs -t %{install-root}/%{bindir}

build-depends:
- base.bst
