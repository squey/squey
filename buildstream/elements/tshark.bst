kind: cmake
description: |

    TShark is a terminal oriented version of Wireshark designed for capturing and displaying packets
    when an interactive user interface isn’t necessary or available

sources:
- kind: tar
  url: gitlab:wireshark/wireshark/-/archive/wireshark-4.4.9/wireshark-wireshark-4.4.9.tar.bz2
  ref: a476c3435bbd5254840ad8c860cf55c853ac9c955f220984813c2ac677735cb5

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
