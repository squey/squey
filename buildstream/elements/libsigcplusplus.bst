kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual. 

sources:
- kind: tar

  url: gnome:sources/libsigc++/3.4/libsigc%2B%2B-3.4.0.tar.xz
  ref: 02e2630ffb5ce93cd52c38423521dfe7063328863a6e96d41d765a6116b8707e

depends:
- base.bst
