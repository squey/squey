kind: meson
description: |

    msitools is a set of programs to inspect and build Windows Installer (.MSI) files.
    It is based on libmsi, a portable library to read and write .MSI files.
    libmsi in turn is a port of (and a subset of) Wine`s implementation of the Windows Installer. 

sources:
- kind: tar
  url: gnome:sources/msitools/0.103/msitools-0.103.tar.xz
  ref: d17622eebbf37fa4c09b59be0bc8db08b26be300a6731c74da1ebce262bce839

environment:
  XDG_DATA_DIRS: "%{prefix}/share"

depends:
- msi_package/libgfs.bst
- msi_package/gcab.bst
- freedesktop-sdk.bst:components/libxml2.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/gobject-introspection.bst

build-depends:
- base.bst
- freedesktop-sdk.bst:components/meson.bst
- freedesktop-sdk.bst:components/vala.bst
- freedesktop-sdk.bst:components/bison.bst