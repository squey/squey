kind: cmake
description: |

    Library for general-purpose tracing based on PcapNg file format.
    The current project represents work in progress to provide an API
    that can read, write and manipulate PcapNg files (possibly outside
    normal contexts).

sources:
- kind: git
  url: https://github.com/rvelea/LightPcapNg.git
  ref: ff286c2b89a6541ef1793a7f931fa0f94206d26f
  
config:
  install-commands:
    - mkdir -p "%{install-root}%{prefix}/include" "%{install-root}%{prefix}/lib"
    - cp include/* "%{install-root}%{prefix}/include"
    - find . -name "liblight_pcapng_static.a" -exec cp "{}" "%{install-root}%{prefix}/lib" \;
    
depends:
- base.bst



 
