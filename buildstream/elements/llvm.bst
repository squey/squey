kind: import
description: |

    Import the llvm freedesktop SDK extension

config:
  source: files
  target: usr

sources:
    - kind: ostree
      url: flathub:repo/
      track: org.freedesktop.Sdk.Extension.llvm22/x86_64/26.08
      ref: dce3c1958ec49f0c8b5934ed671e938e1992febb1738c6bb8ebb28770c71d70d
