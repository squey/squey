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
      track: org.freedesktop.Sdk.Extension.mingw-w64/x86_64/26.08
      ref: 8ba08ed5a47dfe1757797fec1042b67d0bcb025984484998b9d381bfd300b748

depends:
- base.bst
- freedesktop-sdk.bst:components/gmp.bst
# The GCC 16 this extension ships is linked against libmpfr.so.6, which used to
# be pulled in by the older toolchain: without it cc1 cannot even start.
- freedesktop-sdk.bst:components/mpfr.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/flex.bst
