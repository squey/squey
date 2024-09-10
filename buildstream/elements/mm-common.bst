kind: meson
description: |

    The mm-common module provides the build infrastructure and utilities
    shared among the GNOME C++ binding libraries.  It is only a required
    dependency for building the C++ bindings from the gnome.org version
    control repository.  An installation of mm-common is not required for
    building tarball releases, unless configured to use maintainer-mode.

sources:
- kind: tar
  url: gnome:sources/mm-common/1.0/mm-common-1.0.6.tar.xz
  ref: b55c46037dbcdabc5cee3b389ea11cc3910adb68ebe883e9477847aa660862e7

depends:
- base.bst

build-depends:
- freedesktop-sdk.bst:components/meson.bst