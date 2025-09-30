kind: autotools

sources:
- kind: git_repo
  url: github:curl/curl.git
  track: curl-*
  ref: curl-8_16_0-0-g11b991232fbcaa88e2b1faecac224416b0001e35

(@): buildstream/elements/base/target.yml

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
