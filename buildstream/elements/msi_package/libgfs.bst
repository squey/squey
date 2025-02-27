kind: autotools
description: |

    libgfs aims to provide an efficient extensible i/o abstraction for
    dealing with different structured file formats.

sources:
- kind: tar
  url: gnome:sources/libgsf/1.14/libgsf-1.14.53.tar.xz
  ref: 0eb59a86e0c50f97ac9cfe4d8cc1969f623f2ae8c5296f2414571ff0a9e8bcba

build-depends:
- base.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:components/automake.bst
- freedesktop-sdk.bst:components/autoconf.bst
- freedesktop-sdk.bst:components/libxml2.bst

depends:
- freedesktop-sdk.bst:components/glib.bst