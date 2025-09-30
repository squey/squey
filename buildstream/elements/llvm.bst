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
      ref: 735476d3b5061b8d2cf45871b0331f8d0089b04b0db4799bec62927506b5d99e
