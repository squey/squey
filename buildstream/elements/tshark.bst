kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual. 

sources:
- kind: git

  url: https://gitlab.com/wireshark/wireshark
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
