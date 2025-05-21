kind: autotools
description: |

    Mac OS X cross toolchain for Linux, FreeBSD, OpenBSD and Android (Termux) 

sources:
- kind: git  
  url: github:tpoechtrager/osxcross.git
  ref: 29fe6dd35522073c9df5800f8cd1feb4b9a993a8

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
  ref: aa37c11ad1a817248c9d1578ac99e133875b4eb5
  directory: build/apple-libtapi/

- kind: git  
  url: github:tpoechtrager/apple-libdispatch.git
  ref: fdf3fc85a9557635668c78801d79f10161d83f12
  directory: build/apple-libdispatch/

- kind: git  
  url: github:tpoechtrager/cctools-port.git
  ref: 164f42296d56fd9456cb9e969ad6cf95e430cb39
  directory: build/cctools-port/

- kind: tar  
  url: github:llvm/llvm-project/archive/refs/tags/llvmorg-18.1.8.tar.gz
  ref: 09c08693a9afd6236f27a2ebae62cda656eba19021ef3f94d59e931d662d4856
  directory: build/compiler-rt/

- kind: local
  path: buildstream/files/macos_sdk
  directory: tarballs/

config:
  configure-commands:
  - sed 's/git_clone_repository "${url}" "${branch}" "${project_name}"/f_res=0/' -i tools/tools.sh
  - sed 's/tarflags="xf"/tarflags="--no-same-owner -xf"/' -i tools/tools.sh
  - sed '90,94d' -i build_compiler_rt.sh
  - sed '44,45d' -i tools/toolchain.cmake
  
  build-commands:
  - UNATTENDED=1 ./build.sh
  - ./build_compiler_rt.sh

  install-commands:
  - mkdir -p "%{install-root}%{prefix}/lib/clang/18/lib/darwin"
  - mv target "%{install-root}%{prefix}/osxcross"
  - mv build/compiler-rt/compiler-rt/build/lib/darwin/*.a "%{install-root}%{prefix}/lib/clang/18/lib/darwin"
  - mv build/compiler-rt/compiler-rt/build/lib/darwin/*.dylib "%{install-root}%{prefix}/lib/clang/18/lib/darwin"

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