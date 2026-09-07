kind: autotools

sources:
- kind: git_repo
  url: github:PCRE2Project/pcre2.git
  track: pcre2-*
  exclude:
  - '*-RC*'
  ref: pcre2-10.48-0-g7978954dbd2efc6f2196869290553cf1871b4ce6

(@): buildstream/elements/base/target.yml

variables:
  conf-local: "--enable-pcre2-16 --enable-pcre2-32"

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpcre2-8.%{dylib-extension}" "%{install-root}%{libdir}/libpcre2-8.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpcre2-16.%{dylib-extension}" "%{install-root}%{libdir}/libpcre2-16.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpcre2-32.%{dylib-extension}" "%{install-root}%{libdir}/libpcre2-32.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpcre2-posix.%{dylib-extension}" "%{install-root}%{libdir}/libpcre2-posix.%{dylib-extension}"

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
