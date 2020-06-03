kind: manual
description: |

  INENDI Inspector HTML reference manual

sources:
- kind: local
  path: doc
  
environment:
  DOCS: "documents/inendi_inspector_reference_manual"
  INSTALLROOT: "%{install-root}%{prefix}/share/doc/inendi-inspector"
    
config:
  install-commands:
    - make -j1 install
  
depends:
- base.bst
