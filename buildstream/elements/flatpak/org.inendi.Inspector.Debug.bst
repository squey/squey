kind: flatpak_image

depends:
- filename: flatpak/flatpak-debug-image.bst
  type: build

config:
  directory: '%{debugdir}'

  metadata:
    Runtime:
      name: org.inendi.Inspector.Debug

    ExtensionOf:
      ref: org.inendi.Inspector/%{gcc_arch}/%{branch}
