kind: flatpak_image

depends:
- filename: flatpak/flatpak-debug-image.bst
  type: build

config:
  directory: '%{debugdir}'

  metadata:
    Runtime:
      name: org.squey.Squey.Debug

    ExtensionOf:
      ref: org.squey.Squey/%{gcc_arch}/%{branch}
