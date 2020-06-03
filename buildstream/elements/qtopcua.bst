kind: qmake
description: |

    Qt wrapper for existing OPC UA stacks

sources:
- kind: git
  url: https://code.qt.io/qt/qtopcua.git
  ref: 4e75905d01b6a8aac987613cd4798fff1126dd4a
  
- kind: patch
  path: buildstream/patches/qtopcua/qtopcua_secure.patch
  strip-level: 1
  
environment:
  LD_LIBRARY_PATH: "%{prefix}/lib"
  
config:
  configure-commands:
    - qmake DEFINES+=UA_ENABLE_ENCRYPTION -- -system-open62541 OPEN62541_PREFIX=%{prefix}
    
depends:
- base.bst
- freedesktop-sdk.bst:components/pkg-config.bst
- qt5.bst
- mbedtls.bst
- open62541.bst
