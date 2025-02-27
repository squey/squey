kind: autotools

sources:
- kind: git_repo
  url: github:openssl/openssl.git
  track: refs/tags/openssl-3.*
  exclude:
  - openssl*alpha*
  - openssl*beta*
  ref: openssl-3.3.2-0-gfb7fab9fa6f4869eaa8fbb97e0d593159f03ffe4

(@): buildstream/elements/base/target.bst

config:
  configure-commands:
  - |
    ./Configure \
      %{openssl-target} %{openssl-arch-conf} \
      --prefix=%{prefix} \
      --libdir=%{lib} \
      --openssldir=%{sysconfdir}/pki/tls \
      --cross-compile-prefix="%{toolchain-dir}/%{host}-" \
      shared \
      threads

  (?):
  - (target_triple == "x86_64-linux-gnu"):
      install-commands:
        (>):
        - rm %{install-root}%{libdir}/lib*.a
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libssl.3.%{dylib-extension}" "%{install-root}/%{libdir}/libssl.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -change "%{libdir}/libcrypto.3.%{dylib-extension}" "@rpath/libcrypto.3.%{dylib-extension}" "%{install-root}/%{libdir}/libssl.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libcrypto.3.%{dylib-extension}" "%{install-root}/%{libdir}/libcrypto.%{dylib-extension}"
  - (target_triple == "x86_64-w64-mingw32"):
      install-commands:
        (>):
        - cp -r ms/* "%{install-root}/%{includedir}/openssl"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/c_rehash'
        - '%{libdir}/libssl.%{dylib-extension}'
        - '%{libdir}/libcrypto.%{dylib-extension}'
        - '%{prefix}/ssl/misc/*'
  cpe:
    vendor: 'openssl'

depends:
- sdk.bst

