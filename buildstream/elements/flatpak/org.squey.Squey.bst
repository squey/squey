kind: flatpak_image

depends:
- filename: flatpak/flatpak-image.bst
  type: build
- filename: flatpak/org.squey.Squey_export.bst
  type: build

config:
  directory: '%{prefix}'

  # buildstream/scripts/export_flatpak.sh writes the metadata, from
  # buildstream/files/flatpak/metadata.sh: flatpak_image only has to know that the
  # build is an application.
  metadata:
    Application: {}
