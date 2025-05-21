kind: cmake
description: |

   XZ Utils provide a general-purpose data-compression library plus command-line tools.

sources:
- kind: tar
  url: github:tukaani-project/xz/archive/refs/tags/v5.8.0.tar.gz
  ref: f50fe3279c276b46b1073b4fa5695f02d5a6a7aad38646b795a0d7c01cc10802

(@): buildstream/elements/base/target.bst

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

