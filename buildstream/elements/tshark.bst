kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual. 

sources:
- kind: tar

  url: https://www.wireshark.org/download/src/wireshark-3.2.2.tar.xz
  ref: 5f5923ef4c3fee370ed0ca1bb324f37c246015eba4a7e74ab95d9208feeded79

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
