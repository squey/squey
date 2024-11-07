kind: autotools

sources:
- kind: git_repo
  url: github:PCRE2Project/pcre2.git
  track: pcre2-*
  exclude:
  - '*-RC*'
  ref: pcre2-10.44-0-g6ae58beca071f13ccfed31d03b3f479ab520639b

(@): buildstream/elements/base/target.bst

variables:
  conf-local: "--enable-pcre2-16 --enable-pcre2-32"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/pcre2-config'
        - '%{libdir}/libpcre2-8.%{dylib-extension}'
        - '%{libdir}/libpcre2-16.%{dylib-extension}'
        - '%{libdir}/libpcre2-32.%{dylib-extension}'
        - '%{libdir}/libpcre2-posix.%{dylib-extension}'

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/glib.bst