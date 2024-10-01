kind: cmake
description: |

    TShark is a terminal oriented version of Wireshark designed for capturing and displaying packets
    when an interactive user interface isn’t necessary or available

sources:
- kind: tar
  url: wireshark:src/wireshark-4.4.0.tar.xz
  ref: ead5cdcc08529a2e7ce291e01defc3b0f8831ba24c938db0762b1ebc59c71269

variables:
  cmake-local: |
    -DBUILD_wireshark=OFF \
    -DBUILD_sharkd=OFF \
    -DBUILD_logray=OFF

depends:
- base.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/libgcrypt.bst
- libmaxminddb.bst
