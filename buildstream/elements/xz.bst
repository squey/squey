kind: cmake
description: |

   XZ Utils provide a general-purpose data-compression library plus command-line tools.

sources:
- kind: tar
  url: github:tukaani-project/xz/archive/refs/tags/v5.8.3.tar.gz
  ref: 8ec1767fa517642ecb4cf08b891ce667ba6f143551e382b07c7ef437bda335e2

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
