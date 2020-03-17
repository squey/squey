kind: autotools
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual. 

sources:
- kind: tar

  url: https://download.gnome.org/sources/libsigc++/2.10/libsigc++-2.10.0.tar.xz
  ref: f843d6346260bfcb4426259e314512b99e296e8ca241d771d21ac64f28298d81

config:
  configure-commands:
  - "%{configure} --disable-documentation"

depends:
- base.bst
