kind: autotools
description: GNU Binutils

sources:
  - kind: tar
    url: sourceware:pub/binutils/releases/binutils-2.47.tar.xz
    ref: 154ab23b60070e8f27013c22977f1129425d67d1e8acd6e13010e617811e4cff

(@):
  - buildstream/elements/base/target.yml

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
  # --with-system-zlib links the zlib of the platform the tools run on: the
  # freedesktop SDK's on Linux (the zlib.bst copy is then only a second
  # libz.so.1 under /app, which the runtime search path of everything built in
  # the sandbox would pick over /usr's), the cross-built one on the other targets.
  - (target_triple == "x86_64-linux-gnu"):
      depends:
        (>):
        - freedesktop-sdk.bst:bootstrap/zlib-ng.bst
  - (target_triple != "x86_64-linux-gnu"):
      depends:
        (>):
        - zlib.bst

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
