kind: autotools
description: |

    Qt is the faster, smarter way to create innovative devices, modern UIs & applications for multiple screens. Cross-platform software development at its best.

sources:
- kind: tar

  url: https://download.qt.io/archive/qt/5.13/5.13.1/single/qt-everywhere-src-5.13.1.tar.xz
  ref: adf00266dc38352a166a9739f1a24a1e36f1be9c04bf72e16e142a256436974e
  
  # Can't use version 5.14.1 because of this bug : https://bugreports.qt.io/browse/QTBUG-80990
  #url: https://download.qt.io/archive/qt/5.14/5.14.1/single/qt-everywhere-src-5.14.1.tar.xz
  #ref: 6f17f488f512b39c2feb57d83a5e0a13dcef32999bea2e2a8f832f54a29badb8
  
- kind: patch
  path: buildstream/patches/qt5/xcbegl.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/qt5/Permit-to-specify-a-default-QTransform-in-QGraphisSc.patch
  strip-level: 1
  
- kind: patch
  path: buildstream/patches/qt5/Fix-rectangle-intersection-issue-because-of-floating.patch
  strip-level: 1
  
- kind: patch
  path: buildstream/patches/qtwebengine/Fix_compiler_error_SIOCGSTAMP_not_declared.patch
  strip-level: 1
  
- kind: patch
  path: buildstream/patches/qtwebengine/qtwebengine-no-udev.patch
  strip-level: 1

variables:
  conf-local: |
    -prefix %{prefix} \
    -libdir %{libdir} \
    -sysconfdir %{sysconfdir}/xdg \
    -archdatadir %{libdir}/qt5 \
    -datadir %{prefix}/share/qt5 \
    -docdir %{prefix}/share/doc/qt5 \
    -headerdir %{prefix}/include/qt5 \
    -plugindir %{libdir}/qt5/plugins \
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
    -gtk #\
    #-ccache
    
config:
  configure-commands:
    - ./configure %{conf-local}
      
  build-commands:
    - make -j%{max-jobs}

  install-commands:
    - |
      make -j1 INSTALL_ROOT=%{install-root} install
      rm -f "%{install-root}%{libdir}"/*.a


environment:
  OPENSSL_LIBS: "-L/%{prefix}/lib -lssl -lcrypto"
  
depends:
- base.bst
- freedesktop-sdk.bst:bootstrap/linux-headers.bst
- freedesktop-sdk.bst:components/pkg-config.bst
- freedesktop-sdk.bst:bootstrap/openssl.bst
- freedesktop-sdk.bst:components/perl.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/which.bst
- freedesktop-sdk.bst:components/giflib.bst
- freedesktop-sdk.bst:components/glib.bst
- freedesktop-sdk.bst:components/icu.bst
- freedesktop-sdk.bst:components/pcre2.bst
- freedesktop-sdk.bst:components/sqlite.bst
- freedesktop-sdk.bst:components/cups.bst
- freedesktop-sdk.bst:components/dbus.bst
- freedesktop-sdk.bst:components/fontconfig.bst
- freedesktop-sdk.bst:components/gtk3.bst
- freedesktop-sdk.bst:components/harfbuzz.bst
- freedesktop-sdk.bst:components/vulkan.bst
- freedesktop-sdk.bst:components/xcb-util-image.bst
- freedesktop-sdk.bst:components/xcb-util-keysyms.bst
- freedesktop-sdk.bst:components/xcb-util-renderutil.bst
- freedesktop-sdk.bst:components/xcb-util-wm.bst
- freedesktop-sdk.bst:components/xorg-lib-xcb.bst
- freedesktop-sdk.bst:components/libglvnd.bst
- mysql.bst
  
# QWebEngine
- python2.bst
- freedesktop-sdk.bst:components/nss.bst
- freedesktop-sdk.bst:components/libdrm.bst
- freedesktop-sdk.bst:components/xorg-lib-xcursor.bst
- freedesktop-sdk.bst:components/xorg-lib-xtst.bst
- freedesktop-sdk.bst:components/bison.bst
- freedesktop-sdk.bst:components/flex.bst
- freedesktop-sdk.bst:components/gperf.bst

