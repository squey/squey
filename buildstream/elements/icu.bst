kind: autotools

sources:
- kind: tar
  url: github:unicode-org/icu/releases/download/release-75-1/icu4c-75_1-src.tgz
  ref: cb968df3e4d2e87e8b11c49a5d01c787bd13b9545280fc6642f826527618caef
- kind: patch
  path: buildstream/patches/icu/icu-fix-shared-linking.patch

(@): buildstream/elements/base/target.bst

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