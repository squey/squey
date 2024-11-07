kind: cmake
description: |

    A fast JSON parser/generator for C++ with both SAX/DOM style API

sources:
- kind: tar
  url: github:Tencent/rapidjson/archive/232389d4f1012dddec4ef84861face2d2ba85709.tar.gz
  ref: b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806
  
(@): buildstream/elements/base/target.bst

config:
  configure-commands:
    - if [ "%{target-platform}" = "darwin" ] ; then sed "112d" -i CMakeLists.txt; fi
    - "%{cmake}"

depends:
- sdk.bst