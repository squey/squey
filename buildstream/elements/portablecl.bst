kind: cmake
description: |

    PoCL is a portable open source (MIT-licensed) implementation of the OpenCL standard.

sources:
- kind: tar
  url: github:pocl/pocl/archive/refs/tags/v6.0.tar.gz
  ref: de9710223fc1855f833dbbf42ea2681e06aa8ec0464f0201104dc80a74dfd1f2

(@): buildstream/elements/base/target.bst

environment:
  VERBOSE: 1

variables:
  cmake-local-distro: ''
  cmake-local-linux: ''
  cmake-local-darwin-aarch64: ''
  cmake-local: >-
    -DINSTALL_OPENCL_HEADERS=1
    -DCMAKE_BUILD_TYPE=Release
    -DENABLE_TBB_DEVICE=1

  cmake-local-darwin: >-
    -DCLANG="%{toolchain-dir}/%{cc-compiler}"
    -DCLANGXX="%{toolchain-dir}/%{cxx-compiler}"
    -DCMAKE_C_FLAGS="-L%{sdk-dir}/usr/lib -isystem %{sdk-dir}/usr/include"
    -DCMAKE_CXX_FLAGS="-L%{sdk-dir}/usr/lib -isystem %{sdk-dir}/usr/include"
    -DBUILD_PREFIX="%{host-prefix}"
    -DHOST_PREFIX="%{prefix}"
    -DLLVM_HOST_TARGET="%{host}"
    -DTARGET_METAL=ON
    -DCMAKE_SYSROOT="%{prefix}"
    -DLLVM_BINDIR="%{host-prefix}/bin"
    -DLLVM_LIBDIR="%{libdir}"
    -DSTATIC_LLVM=OFF
    -DDL_H="%{sdk-dir}/usr/include/dlfcn.h"
    -DEXTRA_HOST_LD_FLAGS="-L@loader_path -L@loader_path/.."

  (?):
  - (target_triple == "x86_64-linux-gnu" or target_triple == "x86_64-apple-darwin"):
      cmake-local-distro: >-
        -DKERNELLIB_HOST_CPU_VARIANTS=distro
  - target_triple == "aarch64-apple-darwin":
      cmake-local-darwin-aarch64: >-
        -DLLC_TRIPLE="%{host}"
        -DLLC_HOST_CPU="apple-m1"
        -DHOST_DEVICE_BUILD_HASH="cpu-aarch64-apple-macosx10.13.0-apple-m1"
  - target_triple == "x86_64-linux-gnu":
      cmake-local-linux: >-
        -DEXTRA_HOST_LD_FLAGS="-L%{libdir}"

config:
  configure-commands:
  - |
    if [ "%{target-platform}" = "darwin" ] ; then
      sed '937s/.*/    set(DEFAULT_HOST_LD_FLAGS "-dynamiclib -w -v")/' -i CMakeLists.txt
      sed '1092s/.*/  set(LIBMATH)/' -i CMakeLists.txt
      sed '1414s/.*/  {"-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};/' -i lib/CL/devices/common.c
      %{cmake} %{cmake-local-distro} %{cmake-local-darwin} %{cmake-local-darwin-aarch64}
    else
      %{cmake} %{cmake-local-distro} %{cmake-local-linux}
    fi
    
build-depends:
- sdk.bst

depends:
- freedesktop-sdk.bst:components/opencl.bst
- freedesktop-sdk.bst:components/patch.bst
- freedesktop-sdk.bst:components/python3.bst
- tbb.bst
- libhwloc.bst

(?):
  - target_triple == "x86_64-linux-gnu":
      depends:
        (>):
        - clang_host.bst
        - binutils.bst
  - target_triple != "x86_64-linux-gnu":
      depends:
        (>):
        - clang_target.bst


