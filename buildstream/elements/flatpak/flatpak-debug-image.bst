kind: compose

depends:
- filename: squey-cleanup.bst
  type: build

config:
  include-orphans: False
  include:
  - debug
