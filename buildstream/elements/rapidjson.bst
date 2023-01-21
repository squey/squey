kind: cmake
description: |

    A fast JSON parser/generator for C++ with both SAX/DOM style API

sources:
- kind: git

  url: https://github.com/Tencent/rapidjson
  ref: 012be8528783cdbf4b7a9e64f78bd8f056b97e24
  
variables:
  cmake-local: |
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess -Wno-error=implicit-fallthrough="

depends:
- base.bst
