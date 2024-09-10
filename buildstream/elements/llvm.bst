kind: import
description: |

    Import the llvm freedesktop SDK extension

config:
  source: files
  target: usr

sources:
    - kind: ostree
      url: flathub:repo/
      track: org.freedesktop.Sdk.Extension.llvm18/x86_64/24.08
      ref: b4df7e54e59a4121693c0b2bbdf01f1baff078eaa4f53410124f6fdad69596f7