kind: cmake
description: |

    Abseil is an open-source collection of C++ library code designed to augment the C++ standard library. 
    
sources:
- kind: tar
  url: github:abseil/abseil-cpp/archive/refs/tags/20240116.1.tar.gz
  ref: 3c743204df78366ad2eaf236d6631d83f6bc928d1705dd0000b872e53b73dc6a

variables:
  cmake-local: |
    -DCMAKE_CXX_FLAGS="-fPIC"

depends:
- base.bst

