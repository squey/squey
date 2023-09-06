kind: manual

depends:
- base.bst

config:
  install-commands:
  - |
    mkdir -p "%{install-root}%{prefix}/lib/"
    ln -s /usr/lib/x86_64-linux-gnu/libopcodes-2.40.so "%{install-root}%{prefix}/lib/libopcodes-2.39.50.so"
    ln -s /usr/lib/x86_64-linux-gnu/libbfd-2.40.so "%{install-root}%{prefix}/lib/libbfd-2.39.50.so"