kind: cmake
description: |

    TShark is a terminal oriented version of Wireshark designed for capturing and displaying packets
    when an interactive user interface isn’t necessary or available

sources:
- kind: git

  url: gitlab:wireshark/wireshark
  ref: 4a304d7ec2224a766b7c32b37bbff295f51b9e77

variables:
  cmake-local: |
    -DBUILD_wireshark=0

depends:
- base.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/libgcrypt.bst
- libmaxminddb.bst
