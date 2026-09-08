kind: manual

sources:
- kind: remote
  url: github:mikefarah/yq/releases/download/v4.53.6/yq_linux_amd64
  ref: c5f056448f973ae7d39b5401949648a78f2dc1947d6a8eb65be60d5c504b9385

config:
  install-commands:
    - install -D yq_linux_amd64 "%{install-root}%{bindir}/yq"

build-depends:
- base.bst