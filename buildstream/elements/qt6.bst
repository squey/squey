kind: autotools
description: |

    Qt is the faster, smarter way to create innovative devices, modern UIs & applications for multiple screens. Cross-platform software development at its best.

sources:
- kind: tar
  url: qt:archive/qt/6.7/6.7.2/single/qt-everywhere-src-6.7.2.tar.xz
  ref: 0aaea247db870193c260e8453ae692ca12abc1bd841faa1a6e6c99459968ca8a

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
    -xcb-xlib \
    -bundled-xcb-xinput \
    -skip qt3d \
    -skip qtquick3d \
    -skip qtquick3dphysics \
    -skip qtcanvas3d \
    -skip qtandroidextras \
    -skip qtconnectivity \
    -skip qtdatavis3d \
    -skip qtdoc \
    -skip qtlocation \
    -skip qtmacextras \
    -skip qtmultimedia \
    -skip qtopcua \
    -skip qtgamepad \
    -skip qtpurchasing \
    -skip qtsensors \
    -skip qtserialbus \
    -skip qtserialport \
    -skip qtspeech \
    -skip qtvirtualkeyboard \
    -skip qtgraphs \
    -gtk \
    -webengine-icu \
    -no-feature-webengine-geolocation \
    -no-feature-webengine-kerberos \
    -no-feature-webengine-pepper-plugins \
    -no-feature-webengine-printing-and-pdf \
    -no-feature-webengine-spellchecker \
    -no-feature-webengine-webrtc \
    -no-feature-webengine-webrtc-pipewire \
    -no-feature-webengine-developer-build \
    -no-feature-webengine-full-debug-info \
    -no-feature-webengine-native-spellchecker \
    -no-feature-webengine-system-minizip \
    -no-webengine-proprietary-codecs \
    -no-webengine-pulseaudio \
    -no-webengine-alsa \
    -no-webengine-ffmpeg
    
config:
  #create-dev-shm: true

  configure-commands:
    - ./configure %{conf-local}
    - sed -i '18d;19d;21d;' qtwebengine/src/3rdparty/chromium/third_party/minigbm/src/drv.c
      
  build-commands:
    - cmake --build . --parallel

  install-commands:
    - |
      cmake --install . --prefix=%{install-root}%{prefix}


environment:
  OPENSSL_LIBS: "-L/%{prefix}/lib -lssl -lcrypto"
  PYTHONPATH: "/usr/lib/python3.12/site-packages:/%{prefix}/lib/python3.12/site-packages"

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
- freedesktop-sdk.bst:components/freetype.bst
- freedesktop-sdk.bst:components/gtk3.bst
- freedesktop-sdk.bst:components/harfbuzz.bst
- freedesktop-sdk.bst:components/vulkan-headers.bst
- freedesktop-sdk.bst:components/xcb-util.bst
- freedesktop-sdk.bst:components/xcb-util-cursor.bst
- freedesktop-sdk.bst:components/xcb-util-image.bst
- freedesktop-sdk.bst:components/xcb-util-keysyms.bst
- freedesktop-sdk.bst:components/xcb-util-renderutil.bst
- freedesktop-sdk.bst:components/xcb-util-wm.bst
- freedesktop-sdk.bst:components/xorg-lib-xcb.bst
- freedesktop-sdk.bst:components/xorg-lib-xrender.bst
- freedesktop-sdk.bst:components/xorg-lib-xrandr.bst
- freedesktop-sdk.bst:components/xorg-lib-xfixes.bst
- freedesktop-sdk.bst:components/xorg-lib-xext.bst
- freedesktop-sdk.bst:components/xorg-lib-x11.bst
- freedesktop-sdk.bst:components/xorg-lib-xinerama.bst
- freedesktop-sdk.bst:components/libxkbcommon.bst
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
