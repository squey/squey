kind: manual

sources:
- kind: remote
  url: github:mikefarah/yq/releases/download/v4.45.1/yq_linux_amd64
  ref: 654d2943ca1d3be2024089eb4f270f4070f491a0610481d128509b2834870049

config:
  install-commands:
    - install -D yq_linux_amd64 "%{install-root}%{bindir}/yq"

build-depends:
- base.bst