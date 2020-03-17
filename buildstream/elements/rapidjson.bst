kind: cmake
description: |

    A fast JSON parser/generator for C++ with both SAX/DOM style API

sources:
- kind: tar

  url: https://github.com/Tencent/rapidjson/archive/v1.1.0.tar.gz
  ref: bf7ced29704a1e696fbccf2a2b4ea068e7774fa37f6d7dd4039d0787f8bed98e
  
variables:
  cmake-local: |
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess -Wno-error=implicit-fallthrough="

depends:
- base.bst
