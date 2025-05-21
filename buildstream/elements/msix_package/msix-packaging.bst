kind: manual
description: |

    Microsoft MSIX SDK

sources:
- kind: git
  url: github:microsoft/msix-packaging
  ref: 9c9ad98393bee1919fb0ff3a83a1fd52b960ae14
  checkout-submodules: True

config:
  configure-commands:
    - |
      sed -e 's|^cmake -D|cmake -DCMAKE_INSTALL_PREFIX:PATH=%{prefix} -DCMAKE_INSTALL_LIBDIR:PATH=%{lib} -D|' -e 's|^make|make\nDESTDIR=%{install-root} make install|' -i makelinux.sh
      ./makelinux.sh --pack
      mkdir -p "%{install-root}%{bindir}" "%{install-root}%{libdir}"
      cp -r .vs/bin/* "%{install-root}%{bindir}"
      cp -r .vs/lib/* "%{install-root}%{libdir}"

build-depends:
- clang_host.bst

depends:
- base.bst
- msix_package/icu.bst