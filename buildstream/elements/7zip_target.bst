kind: manual
description: |

    7-Zip is a file archiver with a high compression ratio.

(?):
  - target_triple == "x86_64-w64-mingw32":
      sources:
      - kind: remote
        url: github:ip7z/7zip/releases/download/24.09/7z2409-x64.exe
        ref: bdd1a33de78618d16ee4ce148b849932c05d0015491c34887846d431d29f308e
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      sources:
      - kind: remote
        url: github:ip7z/7zip/releases/download/24.09/7z2409-mac.tar.xz
        ref: 073b6fa8dc5d9adb6f742888d0d75f5767406b503199b9704ccbf61133a21ded
  
(@): buildstream/elements/base/target.bst

config:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      install-commands:
      - |
        7zz x 7z2409-x64.exe
        install -D 7z.exe -t %{install-root}/%{bindir}
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
      - |
        tar Jxvf *.tar.xz
        install -D 7zz -t %{install-root}/%{bindir}

build-depends:
- sdk.bst
- 7zip_host.bst