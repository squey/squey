kind: autotools
description: |

    Mac OS X cross toolchain for Linux, FreeBSD, OpenBSD and Android (Termux)

sources:
- kind: git
  url: github:tpoechtrager/osxcross.git
  ref: 27d21e4977c9751d01199c7a226a6faf494c3dd9

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
  ref: fa9443738c1a18accef4244732ec6d6ee97a8133
  directory: build/apple-libtapi/

- kind: git
  url: github:tpoechtrager/apple-libdispatch.git
  ref: 323b9b4e0ca05d6c56a0c2f2d7d8d47363e612b7
  directory: build/apple-libdispatch/

- kind: git
  url: github:tpoechtrager/cctools-port.git
  ref: 904de2a71d4da6a9b30d2efaf912a10ddc7d9ddb
  directory: build/cctools-port/

- kind: tar
  url: github:llvm/llvm-project/archive/refs/tags/llvmorg-22.1.8.tar.gz
  ref: ad18b70e287954c3d62bc7e0b86e7b7af2adf87bcfce21c15fe717f101d7aace
  directory: build/compiler-rt/

- kind: patch
  path: buildstream/patches/osxcross/stable-with-libdispatch.patch
  strip-level: 1

- kind: local
  path: buildstream/files/macos_sdk
  directory: tarballs/

config:
  configure-commands:
  - sed 's/git_clone_repository "${url}" "${branch}" "${project_name}"/f_res=0/' -i tools/tools.sh
  - sed 's/tarflags="xf"/tarflags="--no-same-owner -xf"/' -i tools/tools.sh
  # Keep the compiler-rt sources buildstream staged: upstream wipes any
  # 'compiler-rt' directory that is not one of its own build projects.
  - sed '/^# Check if a build project for compiler-rt already exists\.$/,/^fi$/d' -i build_compiler_rt.sh
  # Let the project-wide pkg-config search path through instead of the macports one.
  - sed '/^set(ENV{PKG_CONFIG_LIBDIR}/d; /^set(ENV{PKG_CONFIG_SYSROOT_DIR}/d' -i tools/toolchain.cmake

  build-commands:
  # cctools' configure now wants libblocksruntime, which comes from
  # apple-libdispatch. Only build.sh's 'latest' flavor builds that, and
  # UNATTENDED=1 selects 'stable', so build it here from the sources staged
  # alongside the others.
  - |
    # clang, like build.sh itself uses: libdispatch passes -fmodule-map-file,
    # which gcc does not know.
    cmake -S build/apple-libdispatch -B build/apple-libdispatch/build \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX="$PWD/target" \
      -DCMAKE_INSTALL_LIBDIR=lib
    make -C build/apple-libdispatch/build install -j$(nproc)
  - UNATTENDED=1 ./build.sh
  - ./build_compiler_rt.sh

  install-commands:
  - mkdir -p "%{install-root}%{prefix}/lib/clang/22/lib/darwin"
  - mv target "%{install-root}%{prefix}/osxcross"
  - mv build/compiler-rt/compiler-rt/build/lib/darwin/*.a "%{install-root}%{prefix}/lib/clang/22/lib/darwin"
  - mv build/compiler-rt/compiler-rt/build/lib/darwin/*.dylib "%{install-root}%{prefix}/lib/clang/22/lib/darwin"

depends:
- base/sdk.bst
- clang_host.bst
- freedesktop-sdk.bst:bootstrap/bzip2.bst
- freedesktop-sdk.bst:bootstrap/xz.bst
- freedesktop-sdk.bst:bootstrap/zlib-ng.bst
- freedesktop-sdk.bst:components/git.bst
- freedesktop-sdk.bst:components/cmake.bst
- freedesktop-sdk.bst:components/patch.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:components/libxml2.bst
- freedesktop-sdk.bst:components/cpio.bst
- freedesktop-sdk.bst:components/llvm.bst
