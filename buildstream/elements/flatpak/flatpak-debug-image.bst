kind: compose

depends:
- filename: inendi-inspector-cleanup.bst
  type: build

config:
  include-orphans: False
  include:
  - debug
