kind: autotools
description: |

    Breakpad is a set of client and server components which implement a crash-reporting system.
  
sources:
- kind: git
  url: gitlab:squey/breakpad
  ref: d6893d1b875da0ef88b6de24f60a9940ffd32380

(@): buildstream/elements/base/target.bst
    
environment:
  CXXFLAGS: "-std=c++20"

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/python3.bst
