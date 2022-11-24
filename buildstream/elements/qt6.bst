kind: autotools
description: |

    Qt is the faster, smarter way to create innovative devices, modern UIs & applications for multiple screens. Cross-platform software development at its best.

sources:
- kind: tar
  url: https://download.qt.io/archive/qt/6.4/6.4.1/single/qt-everywhere-src-6.4.1.tar.xz
  ref: e20b850b6134098a7f2e7701cfddfb213c6cf394b9e848e6fbc5b0e89dcfcc09

- kind: patch
  path: buildstream/patches/qt6/Permit-to-specify-a-default-QTransform-in-QGraphisSc.patch
  strip-level: 1
  
- kind: patch
  path: buildstream/patches/qt6/Fix-rectangle-intersection-issue-because-of-floating.patch
  strip-level: 1

variables:
  conf-local: |
    -prefix %{prefix} \
    -libdir %{libdir} \
    -sysconfdir %{sysconfdir}/xdg \
    -archdatadir %{libdir}/qt6 \
    -datadir %{prefix}/share/qt6 \
    -docdir %{prefix}/share/doc/qt6 \
    -headerdir %{prefix}/include/qt6 \
    -plugindir %{libdir}/qt6/plugins \
    -confirm-license \
    -opensource \
    -release \
    -openssl-linked \
    -sql-mysql \
    -sql-sqlite \
    -platform linux-g++ \
    -opengl es2 \
    -opengles3 \
    -optimized-qmake \
    -nomake examples \
    -nomake tests \
    -accessibility \
    -dbus-linked \
    -fontconfig \
    -glib \
    -system-proxies \
    -no-feature-renameat2 \
    -no-feature-getentropy \
    -no-feature-statx \
    -gtk
    
config:
  create-dev-shm: true

  configure-commands:
    - ./configure %{conf-local}
      
  build-commands:
    - cmake --build . --parallel

  install-commands:
    - |
      cmake --install . --prefix=%{install-root}%{prefix}
      rm -f "%{install-root}%{libdir}"/*.a


environment:
  OPENSSL_LIBS: "-L/%{prefix}/lib -lssl -lcrypto"
  PYTHONPATH: "/usr/lib/python3.10/site-packages:/%{prefix}/lib/python3.10/site-packages"

depends:
- base.bst
- freedesktop-sdk.bst:bootstrap/linux-headers.bst
- freedesktop-sdk.bst:components/pkg-config.bst
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:components/perl.bst
- freedesktop-sdk.bst:components/which.bst
- freedesktop-sdk.bst:components/giflib.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/icu.bst
- freedesktop-sdk.bst:bootstrap/pcre2.bst
- freedesktop-sdk.bst:components/sqlite.bst
- freedesktop-sdk.bst:components/cups.bst
- freedesktop-sdk.bst:components/dbus.bst
- freedesktop-sdk.bst:components/fontconfig.bst
- freedesktop-sdk.bst:components/gtk3.bst
- freedesktop-sdk.bst:components/harfbuzz.bst
- freedesktop-sdk.bst:components/vulkan-headers.bst
- freedesktop-sdk.bst:components/xcb-util-image.bst
- freedesktop-sdk.bst:components/xcb-util-keysyms.bst
- freedesktop-sdk.bst:components/xcb-util-renderutil.bst
- freedesktop-sdk.bst:components/xcb-util-wm.bst
- freedesktop-sdk.bst:components/xorg-lib-xcb.bst
- freedesktop-sdk.bst:components/libglvnd.bst
- mysql.bst
- libpq.bst

# QWebEngine
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/python3-six.bst
- freedesktop-sdk.bst:components/python3-html5lib.bst
- freedesktop-sdk.bst:components/gperf.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/libdrm.bst
- freedesktop-sdk.bst:components/xorg-lib-xcursor.bst
- freedesktop-sdk.bst:components/xorg-lib-xshmfence.bst
- freedesktop-sdk.bst:components/xorg-lib-xtst.bst
- freedesktop-sdk.bst:components/nss.bst
- freedesktop-sdk.bst:components/llvm.bst

build-depends:
- nodejs.bst
- python3-webencodings.bst