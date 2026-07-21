kind: import
description: |

    Import the llvm freedesktop SDK extension

config:
  source: files
  target: usr

sources:
    - kind: ostree
      url: flathub:repo/
      track: org.freedesktop.Sdk.Extension.llvm21/x86_64/25.08
      ref: 28ae7e36dead6805e413a7959689ddf19c7fb9dfd8c4d59c251a34a35fb3088a
