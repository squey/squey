kind: manual
description: |

    7-Zip is a file archiver with a high compression ratio.

(?):
  - target_triple == "x86_64-w64-mingw32":
      sources:
      - kind: remote
        url: github:ip7z/7zip/releases/download/26.03/7z2603-x64.exe
        ref: 0859c524b8a63551848f0c246abddcb1d0b7b656b0fbfe879f8d85e61a9e6edd
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      sources:
      - kind: remote
        url: github:ip7z/7zip/releases/download/26.03/7z2603-mac.tar.xz
        ref: 5ca87677072c59f5602e5c49baa27d4694bacd2259b4e507f0094249d4281480

(@): buildstream/elements/base/target.yml

config:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      install-commands:
      - |
        7zzs x 7z2603-x64.exe
        install -D 7z.exe -t %{install-root}/%{bindir}
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
      - |
        tar Jxvf *.tar.xz
        install -D 7zz -t %{install-root}/%{bindir}

build-depends:
- sdk.bst
- 7zip_host.bst
