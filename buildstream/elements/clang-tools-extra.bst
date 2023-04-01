kind: import
description: Clang tools extra

sources:
- kind: ostree
  url: https://flathub.org/repo
  track: runtime/org.freedesktop.Sdk.Extension.llvm14/x86_64/22.08
  ref: 0b511a7831204d7a0b307be1eb3b88b47d81cf25a6c5bd9c4d28e5a6dcaf54ef

config:
  source: files
  target: usr

