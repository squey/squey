kind: import
description: |

    Import the mingw-w64 freedesktop SDK extension.

    Mingw-w64 is an advancement of the original mingw.org project, created to support the GCC compiler on Windows systems.
    It has forked it in 2007 in order to provide support for 64 bits and new APIs.

config:
  source: files
  target: usr

sources:
    - kind: ostree
      url: flathub:repo/
      track: org.freedesktop.Sdk.Extension.mingw-w64/x86_64/24.08
      ref: 9415ed9428decea376bc669a4c2154ac570c6aeceb5d21349cdfea1839e1e3ff

depends:
- base.bst
- freedesktop-sdk.bst:components/gmp.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/flex.bst