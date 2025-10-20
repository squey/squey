kind: meson
description: |

    waypipe is a proxy for Wayland clients. It forwards Wayland messages and
    serializes changes to shared memory buffers over a single socket. This makes
    application forwarding similar to ssh -X feasible.

sources:
- kind: tar
  url: freedesktop:mstoeckl/waypipe/-/archive/v0.10.4/waypipe-v0.10.4.tar.bz2
  ref: 4de622de39890912a0242e446b8d401f6fe385977985224f15353d40d6f7f0a3

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
