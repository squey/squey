kind: autotools
description: |

    Mac OS X cross toolchain for Linux, FreeBSD, OpenBSD and Android (Termux)

sources:
- kind: git
  url: github:tpoechtrager/osxcross.git
  ref: f873f534c6cdb0776e457af8c7513da1e02abe59

- kind: git
  url: github:tpoechtrager/xar.git
  ref: 5fa4675419cfec60ac19a9c7f7c2d0e7c831a497
  directory: build/xar/

- kind: git
  url: github:tpoechtrager/pbzx.git
  ref: 2a4d7c3300c826d918def713a24d25c237c8ed53
  directory: build/pbzx/

- kind: git
  url: github:tpoechtrager/apple-libtapi.git
  ref: 640b4623929c923c0468143ff2a363a48665fa54
  directory: build/apple-libtapi/

- kind: git
  url: github:tpoechtrager/apple-libdispatch.git
  ref: 323b9b4e0ca05d6c56a0c2f2d7d8d47363e612b7
  directory: build/apple-libdispatch/

- kind: git
  url: github:tpoechtrager/cctools-port.git
  ref: 6e31355ca5babc708681bd906eaaef86c3ac93be
  directory: build/cctools-port/

- kind: tar
  url: github:llvm/llvm-project/archive/refs/tags/llvmorg-21.1.1.tar.gz
  ref: 5f048351ee63050d7fa45b6a1160768fb222a8d306a89e1344515ef7a4bcd278
  directory: build/compiler-rt/

- kind: local
  path: buildstream/files/macos_sdk
  directory: tarballs/

config:
  configure-commands:
  - sed 's/git_clone_repository "${url}" "${branch}" "${project_name}"/f_res=0/' -i tools/tools.sh
  - sed 's/tarflags="xf"/tarflags="--no-same-owner -xf"/' -i tools/tools.sh
  - sed '93,97d' -i build_compiler_rt.sh
  - sed '44,45d' -i tools/toolchain.cmake

  build-commands:
  - UNATTENDED=1 ./build.sh
  - ./build_compiler_rt.sh

  install-commands:
  - mkdir -p "%{install-root}%{prefix}/lib/clang/21/lib/darwin"
  - mv target "%{install-root}%{prefix}/osxcross"
  - mv build/compiler-rt/compiler-rt/build/lib/darwin/*.a "%{install-root}%{prefix}/lib/clang/21/lib/darwin"
  - mv build/compiler-rt/compiler-rt/build/lib/darwin/*.dylib "%{install-root}%{prefix}/lib/clang/21/lib/darwin"

depends:
- base/sdk.bst
- clang_host.bst
- freedesktop-sdk.bst:bootstrap/bzip2.bst
- freedesktop-sdk.bst:bootstrap/xz.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:components/git.bst
- freedesktop-sdk.bst:components/cmake.bst
- freedesktop-sdk.bst:components/patch.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:components/libxml2.bst
- freedesktop-sdk.bst:components/cpio.bst
- freedesktop-sdk.bst:components/llvm.bst
