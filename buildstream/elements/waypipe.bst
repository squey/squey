kind: meson
description: |

    waypipe is a proxy for Wayland clients. It forwards Wayland messages and
    serializes changes to shared memory buffers over a single socket. This makes
    application forwarding similar to ssh -X feasible.

sources:
- kind: tar  
  url: freedesktop:mstoeckl/waypipe/-/archive/v0.8.6/waypipe-v0.8.6.tar.bz2
  ref: da40de2e02d60c2c34d549e791a9019c1ddf9d79f42bfad0c6cb74f3f6af9b16

depends:
- base.bst

build-depends:
- freedesktop-sdk.bst:components/meson.bst
- freedesktop-sdk.bst:components/libdrm.bst
- freedesktop-sdk.bst:components/libva.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/lz4.bst
