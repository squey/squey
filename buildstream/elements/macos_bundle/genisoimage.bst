
kind: cmake
description: |

     genisoimage is a pre-mastering program to generate ISO9660/Joliet/HFS hybrid filesystems.

sources:
- kind: tar
  url: github:Distrotech/cdrkit/archive/refs/tags/distrotech-cdrkit-1.1.11.tar.gz
  ref: efa75ef352923d188f060d141b1250d491cc73195358929f7f0fc0ac74b5cd9e

- kind: patch
  path: buildstream/patches/genisoimage/add-efi-boot.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/genisoimage/allow_to_set_creation_date.diff
  strip-level: 1

- kind: patch
  path: buildstream/patches/genisoimage/fix_format-security.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/genisoimage/fix_libcap_detection.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/genisoimage/fix-implicit-function-declaration.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/genisoimage/gcc10.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/genisoimage/update_geteltorito.patch
  strip-level: 1

config:
    configure-commands:
      (<):
       - |
         sed '1i cmake_minimum_required(VERSION 3.5)' -i CMakeLists.txt
    build-commands:
    - |
      %{make} genisoimage

depends:
- base.bst
- freedesktop-sdk.bst:components/libcap.bst
