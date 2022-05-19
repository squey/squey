kind: flatpak_image

depends:
- filename: flatpak/flatpak-debug-image.bst
  type: build

config:
  directory: '%{debugdir}'

  metadata:
    Runtime:
      name: com.gitlab.inendi.Inspector.Debug

    ExtensionOf:
      ref: com.gitlab.inendi.Inspector/%{gcc_arch}/%{branch}
