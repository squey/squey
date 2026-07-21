kind: autotools
description: |

    lbzip2 is a free, multi-threaded compression utility with support for bzip2 compressed file format.
    lbzip2 can process standard bz2 files in parallel.

sources:
- kind: tar
  url: github:/kjn/lbzip2/releases/download/v2.5/lbzip2-2.5.tar.bz2
  ref: eec4ff08376090494fa3710649b73e5412c3687b4b9b758c93f73aa7be27555b

- kind: patch
  path: buildstream/patches/lbzip2/lbzip2-2.5-glibc-2.28.patch
  strip-level: 1

(@): buildstream/elements/base/target.yml

depends:
- sdk.bst
- freedesktop-sdk.bst:bootstrap/bootstrap.bst
- freedesktop-sdk.bst:bootstrap/coreutils.bst
- freedesktop-sdk.bst:bootstrap/glibc.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
