kind: autotools

sources:
- kind: git_repo
  url: github:curl/curl.git
  track: curl-*
  ref: curl-8_10_1-0-g7eb8c048470ed2cc14dca75be9c1cdae7ac8498b

(@): buildstream/elements/base/target.bst

environment:
  PREFIX_OPENSSL: "%{prefix}"
  PKG_CONFIG_PATH: "%{bindir}/pkgconfig"

variables:
  conf-local: >-
    --without-ca-bundle
    --with-ca-fallback
    --without-libpsl
    --with-openssl=%{prefix}
    --without-nghttp2
    --enable-optimize
    --disable-warnings
    --disable-curldebug
  conf-cmd: "bash -x %{conf-root}/configure"

config:
  configure-commands:
    (<):
    - |
      set -u
      TAG=$(git describe | cut -d- -f2)
      REF=$(git rev-parse curl-${TAG})
      PKGVER="${TAG//_/.}"
      TIMESTAMP=$(git log -1 --format=%cs "${REF}")
      sed -i \
      -e "/\WLIBCURL_VERSION\W/c #define LIBCURL_VERSION \"${PKGVER}\"" \
      -e "/\WLIBCURL_TIMESTAMP\W/c #define LIBCURL_TIMESTAMP \"${TIMESTAMP}\"" \
      include/curl/curlver.h

  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      build-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libcurl.%{dylib-extension}" "lib/.libs/libcurl.%{dylib-extension}"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/curl-config'
        - '%{libdir}/libcurl.%{dylib-extension}'
  cpe:
    version-match: '(\d+)_(\d+)_(\d+)'

build-depends:
- sdk.bst
- openssl.bst
- freedesktop-sdk.bst:components/git.bst
- freedesktop-sdk.bst:components/pkg-config.bst