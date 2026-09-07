kind: cmake
description: |

    TShark is a terminal oriented version of Wireshark designed for capturing and displaying packets
    when an interactive user interface isn’t necessary or available

sources:
- kind: tar
  url: gitlab:wireshark/wireshark/-/archive/v4.6.8/wireshark-v4.6.8.tar.bz2
  ref: 7157e61b3a8b4f38e33a7447ae6bb5b5469c86d28ba8b5e2bb03e1a6efa5156a

variables:
  cmake-local: >-
    -DBUILD_wireshark=OFF
    -DBUILD_sharkd=OFF
    -DBUILD_logray=OFF

build-depends:
# Wireshark's CMake requires a Python 3 interpreter for its code generators.
# It used to come in through glib, which declared python3 as a runtime
# dependency until freedesktop-sdk 26.08 moved it to build-depends.
- freedesktop-sdk.bst:components/python3.bst

depends:
- base.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/libgcrypt.bst
- libmaxminddb.bst
