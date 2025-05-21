kind: cmake
description: |

    A fast JSON parser/generator for C++ with both SAX/DOM style API

sources:
- kind: tar
  url: github:Tencent/rapidjson/archive/232389d4f1012dddec4ef84861face2d2ba85709.tar.gz
  ref: b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806
  
(@): buildstream/elements/base/target.bst

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      configure-commands:
        (<):
        - sed "112d" -i CMakeLists.txt
  - (target_triple == "x86_64-w64-mingw32"):
      configure-commands:
        (<):
        - sed 's|RapidJSON_INCLUDE_DIRS  "@RapidJSON_INCLUDE_DIR@"|RapidJSON_INCLUDE_DIRS  "%{includedir}/rapidjson"|' -i "RapidJSONConfig.cmake.in"
        - cat RapidJSONConfig.cmake.in

depends:
- sdk.bst