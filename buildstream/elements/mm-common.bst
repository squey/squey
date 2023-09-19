kind: autotools
description: |

    The mm-common module provides the build infrastructure and utilities
    shared among the GNOME C++ binding libraries.  It is only a required
    dependency for building the C++ bindings from the gnome.org version
    control repository.  An installation of mm-common is not required for
    building tarball releases, unless configured to use maintainer-mode.

sources:
- kind: tar
  url: gnome:sources/mm-common/0.9/mm-common-0.9.10.tar.xz
  ref: 16c0e2bc196b67fbc145edaecb5dbe5818386504fe5703de27002d77140fa217

config:
  configure-commands:
  - "%{configure} --disable-documentation"

depends:
- base.bst
