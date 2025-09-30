kind: autotools

sources:
- kind: tar
  url: github:unicode-org/icu/releases/download/release-77-1/icu4c-77_1-src.tgz
  ref: 588e431f77327c39031ffbb8843c0e3bc122c211374485fa87dc5f3faff24061
- kind: patch
  path: buildstream/patches/icu/icu-fix-shared-linking.patch

(@): buildstream/elements/base/target.yml

(?):
- target_triple == "x86_64-w64-mingw32":
    environment:
      CXXFLAGS: "-fext-numeric-literals"

config:
  configure-commands:
  - |
    sed 's/-install_name $(libdir)/-install_name @rpath/' -i source/config/mh-darwin
    mkdir "build"
    cd "build"
    sh ../source/runConfigureICU %{icu-platform}
    make
    cd ..
    ./source/configure %{conf-args} --enable-rpath=yes --with-cross-build=`pwd`/build

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/icu-config'
        - '%{libdir}/libicudata.%{dylib-extension}'
        - '%{libdir}/libicui18n.%{dylib-extension}'
        - '%{libdir}/libicuio.%{dylib-extension}'
        - '%{libdir}/libicutest.%{dylib-extension}'
        - '%{libdir}/libicutu.%{dylib-extension}'
        - '%{libdir}/libicuuc.%{dylib-extension}'
        - '%{libdir}/icu'
        - '%{libdir}/icu/**'
  cpe:
    version-match: '(\d+)_(\d+)'
    product: international_components_for_unicode

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/icu.bst
