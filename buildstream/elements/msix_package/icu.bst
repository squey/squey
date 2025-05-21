kind: autotools
description: |

    ICU - International Components for Unicode

    This specific version is needed by msix-packaging

sources:
- kind: tar
  url: github:unicode-org/icu/releases/download/release-56-1/icu4c-56_1-src.tgz
  ref: 3a64e9105c734dcf631c0b3ed60404531bce6c0f5a64bfe1a6402a4cc2314816
- kind: patch
  path: buildstream/patches/icu/icu-fix-shared-linking.patch

variables:
    build-dir: "_build"

config:
  configure-commands:
  - |
    mkdir "%{build-dir}"
    cd "%{build-dir}"
    sh ../source/runConfigureICU "Linux"
    make
    cd ..
    ./source/configure %{conf-args} --with-cross-build="$(pwd)/%{build-dir}"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/icu-config'
        - '%{libdir}/libicudata.so'
        - '%{libdir}/libicui18n.so'
        - '%{libdir}/libicuio.so'
        - '%{libdir}/libicutest.so'
        - '%{libdir}/libicutu.so'
        - '%{libdir}/libicuuc.so'
        - '%{libdir}/icu'
        - '%{libdir}/icu/**'

  cpe:
    version-match: '(\d+)_(\d+)'
    product: international_components_for_unicode


build-depends:
- base.bst
- freedesktop-sdk.bst:components/python3.bst