kind: autotools
description: |

    lbzip2 is a free, multi-threaded compression utility with support for bzip2 compressed file format.
    lbzip2 can process standard bz2 files in parallel.

sources:
- kind: tar
  url: fossies:linux/privat/old/lbzip2-2.5.tar.gz
  ref: 46c75ee93cc95eedc6005625442b2b8e59a2bef3ba80987d0491f055185650e9
  
- kind: patch
  path: buildstream/patches/lbzip2/lbzip2-2.5-glibc-2.28.patch
  strip-level: 1

(@): buildstream/elements/base/target.bst

depends:
- sdk.bst
- freedesktop-sdk.bst:bootstrap-import.bst
- freedesktop-sdk.bst:bootstrap/coreutils.bst
- freedesktop-sdk.bst:bootstrap/glibc.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
