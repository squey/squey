kind: cmake
description: |

    TShark is a terminal oriented version of Wireshark designed for capturing and displaying packets
    when an interactive user interface isn’t necessary or available

sources:
- kind: tar
  url: gitlab:wireshark/wireshark/-/archive/v4.6.4/wireshark-v4.6.4.tar.bz2
  ref: 51607771f25d80a70a83a1d6086dfcb36876c92b95782788c63f25c057ee2e01

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
