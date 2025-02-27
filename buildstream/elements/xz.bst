kind: autotools
description: |

   XZ Utils provide a general-purpose data-compression library plus command-line tools.

sources:
- kind: tar
  url: github:tukaani-project/xz/archive/refs/tags/v5.6.3.tar.gz
  ref: 1256ba1212249569e572ffe412635becbe4b033145ba30ff24dcb7e9c6c04b27

(@): buildstream/elements/base/target.bst

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/liblzma.%{dylib-extension}" "%{install-root}%{libdir}/liblzma.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -change "%{libdir}/liblzma.5.%{dylib-extension}" "@rpath/liblzma.5.%{dylib-extension}" "%{install-root}%{bindir}/xz"

depends:
- sdk.bst
- freedesktop-sdk.bst:components/po4a.bst

