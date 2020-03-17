kind: flatpak_repo

depends:
- filename: deploy-tools/flatpak.bst
  junction: freedesktop-sdk.bst
  type: build

- filename: flatpak/com.esi_inendi.Inspector.bst
  type: build
- filename: flatpak/com.esi_inendi.Inspector.Debug.bst
  type: build

config:
  arch: '%{gcc_arch}'
  environment:
  - deploy-tools/flatpak.bst
