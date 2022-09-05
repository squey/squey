kind: cmake
description: |

    Qt wrapper for existing OPC UA stacks

sources:
- kind: git
  url: https://github.com/qt/qtopcua.git
  ref: 48f9323eba155319276ed5ca9b59d8d6a3919b42
  
#- kind: patch
#  path: buildstream/patches/qtopcua/qtopcua_secure.patch
#  strip-level: 1
  
environment:
  LD_LIBRARY_PATH: "%{prefix}/lib"

variables:
  cmake-local: |
    -DUA_ENABLE_ENCRYPTION=1 \
    -DOPEN62541_PREFIX==%{prefix}
    
depends:
- base.bst
- freedesktop-sdk.bst:components/pkg-config.bst
- qt6.bst
- mbedtls.bst
- open62541.bst
