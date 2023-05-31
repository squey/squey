kind: flatpak_repo

depends:
- filename: deploy-tools/flatpak.bst
  junction: freedesktop-sdk.bst
  type: build

- filename: flatpak/org.squey.Squey.bst
  type: build
- filename: flatpak/org.squey.Squey.Debug.bst
  type: build

config:
  arch: '%{gcc_arch}'
  environment:
  - deploy-tools/flatpak.bst
