kind: import
description: |

    Import the llvm freedesktop SDK extension

config:
  source: files
  target: usr

sources:
    - kind: ostree
      url: flathub:repo/
      track: org.freedesktop.Sdk.Extension.llvm16/x86_64/23.08
      ref: e326d2ef7245af47da876ee2e9c66c60b1e573a3c599605052a6552c63044b52