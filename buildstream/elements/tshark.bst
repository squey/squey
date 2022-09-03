kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual. 

sources:
- kind: tar

  url: https://www.wireshark.org/download/src/wireshark-3.6.7.tar.xz
  ref: cce10a35caa2f79b73d3e6e4dc5388dd47d216114a550fdeb06ae78da0edb7c5

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
