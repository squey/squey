kind: manual

sources:
  - kind: local
    path: buildstream/files

build-depends:
- base.bst
# - squey-cleanup.bst

config:
  install-commands:
  - bash flatpak/install_desktop_integration.sh "%{install-root}%{prefix}"
