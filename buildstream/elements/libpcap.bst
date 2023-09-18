kind: autotools
description: |

     A portable C/C++ library for network traffic capture.

sources:
- kind: tar

  url: tcpdump:release/libpcap-1.9.0.tar.gz
  ref: 2edb88808e5913fdaa8e9c1fcaf272e19b2485338742b5074b9fe44d68f37019

depends:
- base.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/bison.bst
