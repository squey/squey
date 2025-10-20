kind: cmake
description: |

   XZ Utils provide a general-purpose data-compression library plus command-line tools.

sources:
- kind: tar
  url: github:tukaani-project/xz/archive/refs/tags/v5.8.1.tar.gz
  ref: bdbc23fbf9098843357e71e49685724fda2c320c29cb1b25fd90505f14bb0b3d

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -add_rpath @executable_path/../Frameworks %{install-root}%{bindir}/xz

depends:
- sdk.bst
- freedesktop-sdk.bst:components/po4a.bst
