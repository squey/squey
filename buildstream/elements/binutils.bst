kind: autotools
description: GNU Binutils

sources:
- kind: tar
  url: sourceware:pub/binutils/releases/binutils-2.44.tar.xz
  ref: ce2017e059d63e67ddb9240e9d4ec49c2893605035cd60e92ad53177f4377237

(@): buildstream/elements/base/target.bst

(?):
  - target_triple != "x86_64-w64-mingw32":
      variables:
        cflags: ""
  - target_triple == "x86_64-w64-mingw32":
      variables:
        cflags: "-L%{libdir}"
  - target_triple != "none":
      variables:
        target_flags_exceptions: ''
        make-args: ''
        MAKEINFO: true
        conf-local: >-
          CFLAGS="%{cflags}"
          --disable-nls
          --disable-multilib
          --disable-werror
          --enable-gold
          --enable-ld=default
          --enable-plugins
          --disable-sim
          --disable-readline
          --disable-libdecnumber
          --enable-relro=yes
          --enable-lto
          --with-system-zlib
          --with-sysroot=%{libdir}
          --with-lib-path=%{libdir}
          --enable-ld=default
          --enable-shared
          --disable-gprofng
          --disable-doc
          --disable-gdb
          --disable-gdbserver
          host_configargs='lt_cv_sys_lib_dlsearch_path_spec="%{libdir} /usr/lib/"'
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      variables:
        conf-global: ""

config:
  configure-commands:
    (<):
    - |
      # https://sourceware.org/bugzilla/show_bug.cgi?id=29933
      rm -rf gdb

  install-commands:
    - make -j1 DESTDIR="%{install-root}" install

build-depends:
- freedesktop-sdk.bst:components/automake.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/pkg-config.bst
- freedesktop-sdk.bst:components/gettext.bst
- freedesktop-sdk.bst:components/texinfo.bst

depends:
- sdk.bst
- zlib.bst