kind: flatpak_image

depends:
- filename: flatpak/flatpak-debug-image.bst
  type: build

config:
  directory: '%{debugdir}'

  metadata:
    Runtime:
      name: com.esi_inendi.Inspector.Debug

    ExtensionOf:
      ref: com.esi_inendi.Inspector/%{gcc_arch}/%{branch}
