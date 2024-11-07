kind: manual

(@): buildstream/elements/macos_bundle/config.yml

config:
  install-commands:
  - |
    genisoimage -V "Squey" -D -R -apple -o "%{volume-name}.uncompressed.dmg" "/bundle_root"
    dmg "%{volume-name}.uncompressed.dmg" "%{install-root}/%{dmg-image-name}"

build-depends:
- macos_bundle/app-bundle.bst
- macos_bundle/genisoimage.bst
- macos_bundle/libdmg-hfsplus.bst
- base.bst