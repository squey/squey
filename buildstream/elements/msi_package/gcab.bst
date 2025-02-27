kind: meson
description: |

    A GObject library to create cabinet files

sources:
- kind: tar
  url: gnome-gitlab:GNOME/gcab/-/archive/v1.6/gcab-v1.6.tar.bz2
  ref: f1f690dffe3a8cb17aadda790e21614918d9eb0672c8248d4385e4ec4309faf5

variables:
  meson-local: >-
    -Ddocs=false

build-depends:
- base.bst
- freedesktop-sdk.bst:components/meson.bst
- freedesktop-sdk.bst:components/vala.bst
- freedesktop-sdk.bst:components/gobject-introspection.bst

