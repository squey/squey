kind: cmake
description: |

    libsigc++ implements a typesafe callback system for standard C++.
    It allows you to define signals and to connect those signals to any callback function,
    either global or a member function, regardless of whether it is static or virtual.

sources:
- kind: tar
  url: github:libsigcplusplus/libsigcplusplus/archive/refs/tags/3.8.1.tar.gz
  ref: d168c799c32c35b8a036484875df7ac692ec1ad2d7de7f9630c8b1b38617e41f

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

depends:
- sdk.bst
