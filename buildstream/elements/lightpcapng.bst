kind: cmake
description: |

    Library for general-purpose tracing based on PcapNg file format.
    The current project represents work in progress to provide an API
    that can read, write and manipulate PcapNg files (possibly outside
    normal contexts).

sources:
- kind: git
  url: github:rvelea/LightPcapNg.git
  ref: 33296580096f83a9f17ebe7ea3d2c79977a24471

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

config:
  (?):
    - target_triple == "x86_64-w64-mingw32":
        configure-commands:
          (<):
          - sed -i '30,$ d' CMakeLists.txt
  install-commands:
  - |
    mkdir -p "%{install-root}%{includedir}" "%{install-root}%{prefix}/lib"
    cp include/* "%{install-root}%{includedir}"
    find . -name "liblight_pcapng_static.a" -exec cp "{}" "%{install-root}%{prefix}/lib" \;

depends:
- sdk.bst
