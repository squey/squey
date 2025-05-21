kind: manual
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

sources:
- kind: remote
  url: kjkpub:software/pigz/2.3.1-149/pigz.zip
  ref: ffe226432099342d8cd846c039f30d3db934d02474058c66c07064b25604de9e

(@): buildstream/elements/base/target.bst

(?):
- target_triple == "x86_64-w64-mingw32":
    config:
      install-commands:
      - |
        7zz x pigz.zip
        install -D pigz.exe -t "%{install-root}%{bindir}"

build-depends:
- sdk.bst
- 7zip_host.bst
