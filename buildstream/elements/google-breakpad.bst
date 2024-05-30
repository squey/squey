kind: autotools
description: |

    Breakpad is a set of client and server components which implement a crash-reporting system.
  
sources:
  - kind: local
    path: src/external/breakpad
    
environment:
  CXXFLAGS: "-std=c++20"

depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst
