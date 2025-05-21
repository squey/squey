kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual.

sources:
- kind: tar
  url: github:libsigcplusplus/libsigcplusplus/archive/refs/tags/3.6.0.tar.gz
  ref: bbe81e4f6d8acb41a9795525a38c0782751dbc4af3d78a9339f4a282e8a16c38

(@): buildstream/elements/base/target.bst

depends:
- sdk.bst
