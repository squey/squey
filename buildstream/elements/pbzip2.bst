kind: manual
description: |

    The pbzip2 program is a parallel version of bzip2 for use on shared memory machines.

sources:
- kind: remote
  url: sourceforge:pbzip2-for-windows/PBzip2-win32.zip
  ref: eda5927f18c22c701171bda93a7d25a31c36c61e7f54477c8f2eb5014fcf805b

(@): buildstream/elements/base/target.yml

config:
    configure-commands:
      - 7zzs x PBzip2-win32.zip
    install-commands:
      - install -D PBzip2/pbzip2.exe PBzip2/cygwin1.dll -t "%{install-root}%{bindir}"

build-depends:
- 7zip_host.bst

depends:
- sdk.bst
