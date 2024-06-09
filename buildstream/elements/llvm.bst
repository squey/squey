kind: import
description: |

    Import the llvm freedesktop SDK extension

config:
  source: files
  target: usr

sources:
    - kind: ostree
      url: flathub:repo/
      track: org.freedesktop.Sdk.Extension.llvm17/x86_64/23.08
      ref: e929c16f4d7c12d6535aba0b72b3197097861a04e817b8c3d686ef6bf79ef2e4