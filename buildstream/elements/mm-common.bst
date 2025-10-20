kind: meson
description: |

    The mm-common module provides the build infrastructure and utilities
    shared among the GNOME C++ binding libraries.  It is only a required
    dependency for building the C++ bindings from the gnome.org version
    control repository.  An installation of mm-common is not required for
    building tarball releases, unless configured to use maintainer-mode.

sources:
- kind: tar
  url: gnome:sources/mm-common/1.0/mm-common-1.0.7.tar.xz
  ref: 494abfce781418259b1e9d8888c73af4de4b6f3be36cc75d9baa8baa0f2a7a39

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
