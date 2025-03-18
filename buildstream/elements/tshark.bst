kind: cmake
description: |

    TShark is a terminal oriented version of Wireshark designed for capturing and displaying packets
    when an interactive user interface isn’t necessary or available

sources:
- kind: tar
  url: gitlab:wireshark/wireshark/-/archive/wireshark-4.4.2/wireshark-wireshark-4.4.2.tar.bz2
  ref: ca5224def17866c2167086aba07be99aa646c9ebf711f7a5573982f0d90e7278

variables:
  cmake-local: >-
    -DBUILD_wireshark=OFF
    -DBUILD_sharkd=OFF
    -DBUILD_logray=OFF

depends:
- base.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/libgcrypt.bst
- libmaxminddb.bst

