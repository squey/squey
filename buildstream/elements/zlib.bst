kind: cmake
description: zlib

sources:
- kind: git_repo
  url: github:madler/zlib.git
  track: v*
  ref: v1.3.1-0-g51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf

(@): buildstream/elements/base/target.bst

(?):
- (target_triple == "x86_64-linux-gnu"):
    config:
      install-commands:
        (>):
        - |
          find "%{install-root}" -name "lib*.a" -delete
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libz.1.%{dylib-extension}" "%{install-root}%{libdir}/libz.%{dylib-extension}"
- (target_triple == "x86_64-w64-mingw32"):
    config:
      install-commands:
        (>):
        - |
          cd %{install-root}%{bindir}
          ln -s "libzlib.dll" "libz.dll"
          cd %{install-root}%{libdir}
          ln -s "libzlib.dll.a" "libz.dll.a"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/libz.%{dylib-extension}'
  cpe:
    vendor: zlib

build-depends:
- sdk.bst