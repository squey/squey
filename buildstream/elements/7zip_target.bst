kind: manual
description: |

    7-Zip is a file archiver with a high compression ratio.

(?):
  - target_triple == "x86_64-w64-mingw32":
      sources:
      - kind: remote
        url: github:ip7z/7zip/releases/download/25.01/7z2501-x64.exe
        ref: 78afa2a1c773caf3cf7edf62f857d2a8a5da55fb0fff5da416074c0d28b2b55f
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      sources:
      - kind: remote
        url: github:ip7z/7zip/releases/download/25.01/7z2501-mac.tar.xz
        ref: 26aa75bc262bb10bf0805617b95569c3035c2c590a99f7db55c7e9607b2685e0

(@): buildstream/elements/base/target.yml

config:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      install-commands:
      - |
        7zzs x 7z2501-x64.exe
        install -D 7z.exe -t %{install-root}/%{bindir}
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
      - |
        tar Jxvf *.tar.xz
        install -D 7zz -t %{install-root}/%{bindir}

build-depends:
- sdk.bst
- 7zip_host.bst
