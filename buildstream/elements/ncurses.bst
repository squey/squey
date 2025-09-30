kind: autotools
description: GNU ncurses

sources:
- kind: tar
  url: github:mirror/ncurses/archive/refs/tags/v6.4.tar.gz
  ref: 39893846139518e6c2d00ac1d354d4889f0f394acd44885d70b14eaef4e23e8e

(@): buildstream/elements/base/target.yml

variables:
  conf-local: >-
    --without-cxx
    --enable-warnings
    --enable-assertions
    --enable-exp-win32
    --enable-ext-funcs
    --disable-home-terminfo
    --disable-echo
    --disable-getcap
    --disable-hard-tabs
    --disable-leaks
    --disable-macros
    --disable-overwrite
    --enable-opaque-curses
    --enable-opaque-panel
    --enable-opaque-menu
    --enable-opaque-form
    --enable-sp-funcs
    --enable-term-driver
    --enable-interop
    --disable-termcap
    --enable-database
    --with-progs
    --without-libtool
    --without-debug
    --with-fallbacks=ms-terminal
    --without-manpages
    --host="%{host}"
    --disable-stripping
    --libdir=%{libdir}
    --with-pkg-config="%{bindir}/pkg-config"
    --with-pkg-config-libdir="%{libdir}/pkgconfig"
    --disable-root-access
    --disable-setuid-environ
    --with-shared
    --without-ada
    --without-normal
    --enable-pc-files
    --prefix=%{prefix}
    TIC="/usr/bin/tic"
  make-args: ""

config:
  configure-commands:
  - |
    mkdir ncurses-build &&
    cd ncurses-build &&
    ../configure --disable-widec %{conf-local}

  - |
    mkdir ncursesw-build &&
    cd ncursesw-build &&
    ../configure --enable-widec %{conf-local}

  build-commands:
  - |
    cd ncurses-build && %{make}

  - |
    cd ncursesw-build && %{make}

  install-commands:
  - |
    cd ncurses-build && %{make-install}

  - |
    cd ncursesw-build && %{make-install}

  # - |
  #   find "%{install-root}" -name "lib*.a" -delete

  - |
    %{delete-libtool-archives}

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/ncurses6-config'
        - '%{bindir}/ncursesw6-config'
        - '%{libdir}/libtinfo.%{dylib-extension}'
        - '%{libdir}/libtinfow.%{dylib-extension}'
        - '%{libdir}/libformw.%{dylib-extension}'
        - '%{libdir}/libform.%{dylib-extension}'
        - '%{libdir}/libpanel.%{dylib-extension}'
        - '%{libdir}/libmenuw.%{dylib-extension}'
        - '%{libdir}/libmenu.%{dylib-extension}'
        - '%{libdir}/libcurses.%{dylib-extension}'
        - '%{libdir}/libncursesw.%{dylib-extension}'
        - '%{libdir}/libncurses.%{dylib-extension}'
        - '%{libdir}/libpanelw.%{dylib-extension}'

depends:
- sdk.bst
