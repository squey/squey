kind: manual
description: |

    All dependencies to create Squey DMG package (application file layout and build tools)

    This element is used by "make-dmg-image.sh" script within "bst shell" to avoid caching the artifact

(@): buildstream/elements/macos_bundle/config.yml

sources:
- kind: local
  path: buildstream/files/macos_bundle
  directory: buildstream/files/macos_bundle
- kind: local
  path: buildstream/elements/macos_bundle/config.yml
  directory: buildstream/files/macos_bundle
- kind: local
  path: VERSION.txt

build-depends:
- squey-cleanup.bst
- macos_bundle/app-bundle.bst
- macos_bundle/genisoimage.bst
- macos_bundle/libdmg-hfsplus.bst
- freedesktop-sdk.bst:components/zip.bst
- base.bst