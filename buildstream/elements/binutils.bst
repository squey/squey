kind: autotools
description: GNU Binutils

sources:
- kind: tar
  url: sourceware:pub/binutils/releases/binutils-2.43.tar.xz
  ref: b53606f443ac8f01d1d5fc9c39497f2af322d99e14cea5c0b4b124d630379365

variables:
  target_flags_exceptions: ''
  MAKEINFO: true
  conf-local: >-
    CFLAGS="${CFLAGS}"
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
    --with-debuginfod
    --disable-gprofng
    --disable-doc
    --disable-gdb
    --disable-gdbserver
    host_configargs='lt_cv_sys_lib_dlsearch_path_spec="%{libdir} /usr/lib/"'

config:
  configure-commands:
    (<):
    - |
      # https://sourceware.org/bugzilla/show_bug.cgi?id=29933
      rm -rf gdb

  install-commands:
    - make -j1 DESTDIR="%{install-root}" install

build-depends:
- freedesktop-sdk.bst:bootstrap/build/gcc-stage2.bst
- freedesktop-sdk.bst:bootstrap/build/debug-utils.bst
- freedesktop-sdk.bst:bootstrap/gnu-config.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:bootstrap/elfutils.bst
- freedesktop-sdk.bst:components/automake.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/pkg-config.bst
- freedesktop-sdk.bst:components/gettext.bst
- freedesktop-sdk.bst:components/texinfo.bst


depends:
- base.bst