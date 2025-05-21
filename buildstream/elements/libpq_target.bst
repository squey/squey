kind: autotools

(@): 
  - buildstream/elements/libpq.yml
  - buildstream/elements/base/target.bst

environment:
  LDFLAGS: "-L%{libdir}"
  CPPFLAGS: "-I%{includedir}"

depends:
- sdk.bst
- openssl.bst
- icu.bst
- lz4.bst

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpgtypes.%{dylib-extension}" "%{install-root}%{libdir}/libpgtypes.%{dylib-extension}"
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libpq.%{dylib-extension}" "%{install-root}%{libdir}/libpq.%{dylib-extension}"
- (target_triple == "x86_64-w64-mingw32"):
    depends:
      (>):
      - zlib.bst