kind: cmake
description: |

     A portable C/C++ library for network traffic capture.

sources:
- kind: tar
  url: tcpdump:release/libpcap-1.10.5.tar.xz
  ref: 84fa89ac6d303028c1c5b754abff77224f45eca0a94eb1a34ff0aa9ceece3925

(@): buildstream/elements/base/target.bst

variables:
  cmake-local:
    -DPCAP_TYPE=null

(?):
- (target_triple == "x86_64-w64-mingw32"):
    depends:
      (>):
      - openssl.bst

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpcap.A.%{dylib-extension}" "%{install-root}%{libdir}/libpcap.A.%{dylib-extension}"

depends:
- sdk.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/bison.bst
