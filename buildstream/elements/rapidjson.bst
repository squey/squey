kind: cmake
description: |

    A fast JSON parser/generator for C++ with both SAX/DOM style API

sources:
- kind: tar
  url: github:Tencent/rapidjson/archive/24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz
  ref: 2d2601a82d2d3b7e143a3c8d43ef616671391034bc46891a9816b79cf2d3e7a8

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DRAPIDJSON_BUILD_EXAMPLES=FALSE
    -DRAPIDJSON_BUILD_DOC=FALSE

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      configure-commands:
        (<):
        - sed "105d" -i CMakeLists.txt
  - (target_triple == "x86_64-w64-mingw32"):
      configure-commands:
        (<):
        - sed 's|RapidJSON_INCLUDE_DIRS  "@RapidJSON_INCLUDE_DIR@"|RapidJSON_INCLUDE_DIRS  "%{includedir}/rapidjson"|' -i "RapidJSONConfig.cmake.in"
        - cat RapidJSONConfig.cmake.in

depends:
- sdk.bst
