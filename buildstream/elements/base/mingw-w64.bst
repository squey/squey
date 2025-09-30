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
      track: org.freedesktop.Sdk.Extension.mingw-w64/x86_64/25.08
      ref: 25d303a18f3a25b39c18aa2f2aededc04a09ea953a77be0d7bc6dafcec4e1286

depends:
- base.bst
- freedesktop-sdk.bst:components/gmp.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/flex.bst
