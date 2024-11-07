kind: autotools
description: zlib

sources:
- kind: git_repo
  url: github:madler/zlib.git
  track: v*
  ref: v1.3.1-0-g51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf

(@): buildstream/elements/base/target.bst

environment:
  CHOST: "%{toolchain-dir}/%{host}"
  CC: "%{cc-compiler}"
  CXX: "%{cxx-compiler}"
  LD: "%{toolchain-dir}/%{host}-ld"
  RANLIB: "%{toolchain-dir}/%{host}-ranlib"
  AR: "%{toolchain-dir}/%{host}-libtool"
  AS: "%{toolchain-dir}/%{host}-as"
  LIBTOOL: "%{toolchain-dir}/%{host}-libtool"
  STRIP: "%{toolchain-dir}/%{host}-strip"
  NM: "%{toolchain-dir}/%{host}-nm"

variables:
  make-args: ""
  conf-args: >-
    --prefix=%{prefix}
    --sharedlibdir=%{libdir}
    --libdir=%{libdir}

config:
  install-commands:
    (>):
    - |
      find "%{install-root}" -name "lib*.a" -delete

  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libz.1.%{dylib-extension}" "%{install-root}%{libdir}/libz.%{dylib-extension}"

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