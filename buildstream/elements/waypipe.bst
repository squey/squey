kind: meson
description: |

    waypipe is a proxy for Wayland clients. It forwards Wayland messages and
    serializes changes to shared memory buffers over a single socket. This makes
    application forwarding similar to ssh -X feasible.

sources:
- kind: tar
  url: freedesktop:mstoeckl/waypipe/-/archive/v0.11.2/waypipe-v0.11.2.tar.bz2
  ref: be2d59871a9dc67a78f800f08cfba08bb22d660a97a2def24a4df9b8da6196a7

variables:
  meson-local: |
    -Dbuild_c=true -Dbuild_rs=false

depends:
- base.bst

build-depends:
- freedesktop-sdk.bst:components/meson.bst
- freedesktop-sdk.bst:components/libdrm.bst
- freedesktop-sdk.bst:components/libva.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/lz4.bst
