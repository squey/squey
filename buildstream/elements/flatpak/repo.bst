kind: flatpak_repo

depends:
- filename: deploy-tools/flatpak.bst
  junction: freedesktop-sdk.bst
  type: build

- filename: flatpak/org.inendi.Inspector.bst
  type: build
- filename: flatpak/org.inendi.Inspector.Debug.bst
  type: build

config:
  arch: '%{gcc_arch}'
  environment:
  - deploy-tools/flatpak.bst
