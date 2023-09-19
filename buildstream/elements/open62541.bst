kind: cmake
description: |

    open62541 is an open source C (C99) implementation of OPC UA licensed under the Mozilla Public License v2.0.

sources:
- kind: git
  url: github:open62541/open62541.git
  ref: 1b7e2b5d669661c568fda5577ee92011a029cf75
  submodules:
    deps/mdnsd:
      checkout: true
      url: github:Pro/mdnsd.git
    deps/ua-nodeset:
      checkout: true
      url: github:OPCFoundation/UA-Nodeset

variables:
  cmake-local: |
    -DUA_ENABLE_AMALGAMATION=ON \
    -DUA_ENABLE_ENCRYPTION=ON \
    -DUA_ENABLE_HISTORIZING=ON \
    -DUA_ENABLE_EXPERIMENTAL_HISTORIZING=ON \
    -DUA_ENABLE_JSON_ENCODING=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DUA_BUILD_EXAMPLES=OFF \
    -DUA_BUILD_UNIT_TESTS=OFF \
    -DUA_BUILD_TOOLS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DUA_LOGLEVEL=300
    
config:
  install-commands:
    - "%{make-install}"
    - rm -rf "%{install-root}%{prefix}/share"

depends:
- base.bst
- mbedtls.bst
- freedesktop-sdk.bst:components/python3.bst
