kind: autotools
description: |

     A portable C/C++ library for network traffic capture.

sources:
- kind: tar
  url: tcpdump:release/libpcap-1.10.5.tar.xz
  ref: 84fa89ac6d303028c1c5b754abff77224f45eca0a94eb1a34ff0aa9ceece3925

depends:
- base.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/bison.bst
