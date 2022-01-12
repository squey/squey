kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual. 

sources:
- kind: tar

  url: https://www.wireshark.org/download/src/wireshark-3.6.1.tar.xz
  ref: 0434eda8fb6bf88e2b42a67eb5d1de254a67d505bec3bb51fee9d7cad7925a38

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
