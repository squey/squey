kind: meson
description: |

    The mm-common module provides the build infrastructure and utilities
    shared among the GNOME C++ binding libraries.  It is only a required
    dependency for building the C++ bindings from the gnome.org version
    control repository.  An installation of mm-common is not required for
    building tarball releases, unless configured to use maintainer-mode.

sources:
- kind: tar
  url: gnome:sources/mm-common/1.0/mm-common-1.0.8.tar.xz
  ref: b189ee636e839d12c00dabefae099fd488ab2358dec24d264761c011950b02a9

(@): buildstream/elements/base/target.yml


(?):
  - target_triple != "x86_64-linux-gnu":
      variables:
        meson-local: >-
          --cross-file %{host}.txt

config:
  configure-commands:
  - |
    cat <<EOF > "%{host}.txt"
    [binaries]
    name = '%{target-platform}'
    c = '%{cc-compiler}'
    c_ld = '%{toolchain-dir}/%{host}-ld'
    ar = '%{host}-ar'
    ranlib = '%{host}-ranlib'
    strip = '%{host}-strip'

    [host_machine]
    system = '%{target-platform}'
    cpu_family = '%{target-arch}'
    cpu = '%{target-arch}'
    endian = 'little'
    EOF
    %{meson}

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/meson.bst
