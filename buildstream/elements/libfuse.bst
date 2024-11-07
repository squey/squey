kind: autotools
description: |

    The reference implementation of the Linux FUSE (Filesystem in Userspace) interface 

sources:
- kind: tar
  url: github:libfuse/libfuse/archive/refs/tags/fuse-2.9.9.tar.gz
  ref: e57a24721177c3b3dd71cb9239ca46b4dee283db9388d48f7ccd256184982194

- kind: patch
  path: buildstream/patches/libfuse/fuse-2.9.9-closefrom-glibc-2-34.patch

environment:
    MOUNT_FUSE_PATH: "%{prefix}/sbin"

config:
  configure-commands:
  - |
    ./makeconf.sh
    %{configure}

build-depends:
- base.bst