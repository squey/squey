kind: meson
description: |

    waypipe is a proxy for Wayland clients. It forwards Wayland messages and
    serializes changes to shared memory buffers over a single socket. This makes
    application forwarding similar to ssh -X feasible.

sources:
- kind: tar  
  url: freedesktop:mstoeckl/waypipe/-/archive/v0.9.1/waypipe-v0.9.1.tar.bz2
  ref: d60d94a19038d2e231e3f1bf8122ae0894bc78fa753190f6e831c7931f8caaab

depends:
- base.bst

build-depends:
- freedesktop-sdk.bst:components/meson.bst
- freedesktop-sdk.bst:components/libdrm.bst
- freedesktop-sdk.bst:components/libva.bst
- freedesktop-sdk.bst:components/zstd.bst
- freedesktop-sdk.bst:components/lz4.bst
