kind: manual
description: |

    Breakpad is a set of client and server components which implement a crash-reporting system.

sources:
- kind: git
  url: https://chromium.googlesource.com/chromium/tools/depot_tools.git
  ref: befc2993d54938d69d3ec1b707031da5dacf89d4

config:
  configure-commands:
  - |
    ../fetch --no-history breakpad
    cd src && ./configure
  build-commands:
  - make -j${MAXJOBS}
    
variables:
  command-subdir: breakpad

depends:
- base.bst
- python2.bst
