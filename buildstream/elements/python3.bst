kind: autotools

sources:
- kind: git
  url: github:msys2-contrib/cpython-mingw
  ref: 1b241aa8572ee8cd4131fffca838b6bbdf5a7b5e

(@): buildstream/elements/base/target.bst

variables:
  conf-local: >-
    --build=%{target-arch}
    --with-build-python
    --enable-shared
    --with-system-expat
    --with-system-ffi
    --without-ensurepip
    --enable-optimizations
    --with-lto
  local_flags: "-O3"
  version_short: "3"
  version_long: "%{version_short}.12"

config:
  configure-commands:
  - |
    rm -rf configure
    %{autogen}
    %{configure}

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/python3.bst

depends:
- zlib.bst
- xz.bst
- bzip2.bst
- openssl.bst
- libffi.bst
- expat.bst
- ncurses.bst