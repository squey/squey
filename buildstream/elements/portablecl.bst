kind: cmake
description: |

    PoCL is a portable open source (MIT-licensed) implementation of the OpenCL standard.

(@): buildstream/elements/base/target.yml

environment:
  # Was %{bindir}/pkgconfig, a directory nothing ever installs into, which
  # overrode the project-wide setting with an empty search path. pocl looks the
  # ICD loader up through pkg-config, and 'ocl-icd' names OpenCL-Headers in its
  # Requires: with nowhere to find that .pc file, the lookup fails, pocl reports
  # "Requested build with icd, but ICD loader not found" as a mere warning and
  # builds a standalone libOpenCL instead of the libpocl.so an ICD file can
  # point at. opencl-headers.bst installs it under share/pkgconfig, hence both
  # directories here.
  PKG_CONFIG_PATH: "%{libdir}/pkgconfig:%{prefix}/share/pkgconfig:"
  VERBOSE: 1

variables:
  generator: "Unix Makefiles"
  cmake-local-distro: ''
  cmake-local-linux: ''
  cmake-local-darwin-aarch64: ''
  cmake-local-darwin-x86_64: ''
  cmake-local: >-
    -DINSTALL_OPENCL_HEADERS=1
    -DCMAKE_BUILD_TYPE=Release
    -DENABLE_TBB_DEVICE=1

  executable_dir_keyword: "@executable_dir"
  # HOST_CLANG is the osxcross wrapper rather than the bare clang: pocl compiles
  # kernels with it at build time targeting darwin, and only the wrapper passes
  # the -isysroot that makes the macOS headers reachable.
  cmake-local-darwin: >-
    -DCLANG="%{toolchain-dir}/%{cc-compiler}"
    -DCLANGXX="%{toolchain-dir}/%{cxx-compiler}"
    -DCMAKE_C_FLAGS="-L%{sdk-dir}/usr/lib -isystem %{sdk-dir}/usr/include"
    -DCMAKE_CXX_FLAGS="-L%{sdk-dir}/usr/lib -isystem %{sdk-dir}/usr/include"
    -DEXTRA_HOST_LD_FLAGS="-L%{executable_dir_keyword}/../Frameworks -fuse-ld=%{executable_dir_keyword}/ld64.lld"
    -DBUILD_PREFIX="%{host-prefix}"
    -DHOST_PREFIX="%{prefix}"
    -DLLVM_HOST_TARGET="%{host}"
    -DTARGET_METAL=ON
    -DCMAKE_SYSROOT="%{prefix}"
    -DLLVM_BINDIR="%{host-prefix}/bin"
    -DLLVM_LIBDIR="%{libdir}"
    -DWITH_LLVM_CONFIG="%{host-prefix}/bin/llvm-config"
    -DLLVM_DIR="%{libdir}/cmake/llvm"
    -DHOST_CLANG="%{toolchain-dir}/%{cc-compiler}"
    -DHOST_CLANGXX="%{toolchain-dir}/%{cxx-compiler}"
    -DSTATIC_LLVM=OFF
    -DLLC_HOST_CPU_AUTO="x86-64"
    -DDL_H="%{sdk-dir}/usr/include/dlfcn.h"

  # The drivers are linked into pocl.dll rather than dlopen()-ed: as separate
  # DLLs they have to resolve pocl's internal symbols, which pocl.dll does not
  # export since it links with -Wl,--exclude-libs,ALL. ELF targets never hit
  # this because every symbol is visible there by default.
  cmake-local-win: >-
    -DCLANG="/app/bin/clang"
    -DCLANGXX="/app/bin/clang++"
    -DCMAKE_C_FLAGS="-L%{libdir} -isystem %{includedir}"
    -DCMAKE_CXX_FLAGS="-L%{libdir} -isystem %{includedir}"
    -DLLVM_HOST_TARGET="%{host}"
    -DCMAKE_SYSROOT="%{prefix}"
    -DLLVM_BINDIR="%{host-prefix}/bin"
    -DLLVM_LIBDIR="%{libdir}"
    -DWITH_LLVM_CONFIG="%{host-prefix}/bin/llvm-config"
    -DLLVM_DIR="%{libdir}/cmake/llvm"
    -DHOST_CLANG="%{host-prefix}/bin/clang"
    -DHOST_CLANGXX="%{host-prefix}/bin/clang++"
    -DHOST_LLVM_OPT="%{host-prefix}/bin/opt"
    -DHOST_LLVM_LLC="%{host-prefix}/bin/llc"
    -DHOST_LLVM_AS="%{host-prefix}/bin/llvm-as"
    -DHOST_LLVM_DIS="%{host-prefix}/bin/llvm-dis"
    -DHOST_LLVM_LINK="%{host-prefix}/bin/llvm-link"
    -DTARGET_CLANG="%{bindir}/clang.exe"
    -DTARGET_CLANGXX="%{bindir}/clang++.exe"
    -DTARGET_LLVM_OPT="%{bindir}/opt.exe"
    -DTARGET_LLVM_LLC="%{bindir}/llc.exe"
    -DTARGET_LLVM_AS="%{bindir}/llvm-as.exe"
    -DTARGET_LLVM_DIS="%{bindir}/llvm-dis.exe"
    -DTARGET_LLVM_LINK="%{bindir}/llvm-link.exe"
    -DSTATIC_LLVM=OFF
    -DLLC_HOST_CPU_AUTO="x86-64"
    -DENABLE_LOADABLE_DRIVERS=OFF
    -DCMAKE_SIZEOF_VOID_P=8
    -DLLVM_PREFIX="%{prefix}"
    -DLLVM_VERSION_FULL=%{llvm-version}
    -DLLVM_CFLAGS="-I%{includedir}"
    -DLLVM_CXXFLAGS="-I%{includedir}"
    -DLLVM_CPPFLAGS="-I%{includedir}"
    -DLLVM_LDFLAGS="-L%{libdir}"
    -DLLVM_INCLUDE_DIRS="%{includedir}"
    -DLLVM_CMAKEDIR="%{libdir}/cmake/llvm"
    -DLLVM_ALL_TARGETS="X86"
    -DLLVM_HOST_TARGET="x86_64-pc-win32-gnu"
    -DLLVM_BUILD_MODE="Release"
    -DLLVM_ASSERTS_BUILD=OFF
    -DLLVM_HAS_RTTI=YES
    -DLLVM_LIBS="-lLLVMAnalysis -lLLVMBitReader -lLLVMBitWriter -lLLVMCodeGen -lLLVMCore -lLLVMIRReader -lLLVMInstCombine -lLLVMLinker -lLLVMMC -lLLVMObjCARCOpts -lLLVMOption
    -lLLVMPasses -lLLVMScalarOpts -lLLVMSupport -lLLVMTarget -lLLVMTargetParser -lLLVMTransformUtils -lLLVMVectorize -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMX86Desc -lLLVMX86Info -lLLVMipo"
    -DLLVM_SYSLIBS="-luser32 -lshell32 -ladvapi32 -lkernel32 -lws2_32"
    -DLLVM_LLC="/app/bin/llc"
    -DLLVM_AS="/app/bin/llvm-as"
    -DLLVM_OPT="/app/bin/opt"
    -DLLVM_LINK="/app/bin/llvm-link"
    -DLLVM_LLI="/app/bin/lli"
    -DENABLE_PRINTF_IMMEDIATE_FLUSH=OFF
    -DENABLE_TESTS=OFF
    -DENABLE_EXAMPLES=OFF
    -DOPENCL_VERSION="3.0.0"
    -DPOCL_INSTALL_PRIVATE_DATADIR="%{bindir}/pocl"

  commit_ref: "888c9774b94590fff1afef003154ca2c0ea66605"

  (?):
  - (target_triple == "x86_64-linux-gnu" or target_triple == "x86_64-apple-darwin" or target_triple == "x86_64-w64-mingw32"):
      cmake-local-distro: >-
        -DKERNELLIB_HOST_CPU_VARIANTS=distro
  - target_triple == "x86_64-apple-darwin":
      # OCL_KERNEL_TARGET follows LLC_TRIPLE, and pocl falls back to detecting
      # a bare "x86_64-apple-darwin" when it is not told. A triplet with no
      # version reads to clang as a macOS old enough to want -lgcc_s.1 and no
      # -platform_version, so the linker rejects every kernel it is handed and
      # the views stay black. Name the deployment target the rest of the build
      # is compiled against, which is also the triplet the kernel libraries
      # were built under before pocl started guessing.
      cmake-local-darwin-x86_64: >-
        -DLLC_TRIPLE="%{macho-arch}-apple-macosx%{deployment-target-version}.0"
  - target_triple == "aarch64-apple-darwin":
      cmake-local-darwin-aarch64: >-
        -DLLC_TRIPLE="%{host}"
        -DLLC_HOST_CPU="apple-m1"
        -DHOST_DEVICE_BUILD_HASH="%{macho-arch}-apple-darwin%{deployment-target-version}.0"
  - target_triple == "x86_64-linux-gnu":
      cmake-local-linux: >-
        -DEXTRA_HOST_LD_FLAGS="-L%{libdir}"
        -DENABLE_ICD=1

sources:
- kind: git
  url: github:pocl/pocl
  ref: "%{commit_ref}"

- kind: patch
  path: buildstream/patches/portablecl/fix_tbb.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/portablecl/llvm-objcarc-component.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/portablecl/ptrsetwrapper-range.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/portablecl/mingw-icd-exports.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/portablecl/mingw-no-console-window.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/portablecl/macos-lld-platform-version.patch
  strip-level: 1

config:
  (?):
  - target_triple == "x86_64-linux-gnu":
      configure-commands:
      - |
        # pocl links every kernel it compiles against libmvec, glibc's
        # vectorized math library, and since 7.2 it does so by the absolute
        # path cmake resolved rather than -lmvec -- its in-process lld link
        # carries over absolute inputs but drops bare -l flags. That path is
        # the build sysroot's, which carries the whole development symlink
        # farm. The flatpak runtime ships sonames only, so it has libmvec.so.1
        # and no libmvec.so, and every kernel link fails at run time with "no
        # such file or directory" -- leaving the views black.
        #
        # Bake in the link name installed next to pocl below instead, which the
        # application always carries. Patched here rather than through
        # -DLIBMVEC, which cmake link-checks to decide whether libmvec has the
        # vectorized entry points and which cannot point at a path the
        # read-only build sysroot has no room for; and rather than in the
        # generated config.h, which the build regenerates from its template.
        grep -qF 'DEFAULT_HOST_LD_FLAGS} ${LIBMVEC}' CMakeLists.txt || { echo "libmvec link flag not found"; exit 1; }
        sed -i 's|"${DEFAULT_HOST_LD_FLAGS} ${LIBMVEC}"|"${DEFAULT_HOST_LD_FLAGS} %{libdir}/libmvec.so"|' CMakeLists.txt

        %{cmake} %{cmake-local-distro} %{cmake-local-linux}

        grep -qF '%{libdir}/libmvec.so' _builddir/config.h || { echo "libmvec path not baked in"; exit 1; }
      install-commands:
        (>):
        - |
          # The link name the kernel linker was told to use (see above).
          ln -s /usr/lib/x86_64-linux-gnu/libmvec.so.1 "%{install-root}%{libdir}/libmvec.so"
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      configure-commands:
      - |
        sed -E 's/([[:space:]]|")-lm([[:space:]]|")/\1\2/g' -i CMakeLists.txt

        # Patch the lib to look for the linker at runtime using the POCL_LINKER_DIR environment variable
        old="\(device->final_linkage_flags = \)\(final_ld_flags\);"
        grep "$old" lib/CL/devices/common_utils.c || { echo "Patching failed!"; exit 1; }
        sed -e "s/$old/\1update_ld_flags(\2);/" -e '1i #include <executable_dir.h>' -i lib/CL/devices/common_utils.c
        cat <<EOF > "lib/CL/devices/executable_dir.h"
        #include <stdio.h>
        #include <string.h>
        #include <stdlib.h>
        #define EXEC_DIR_KEYWORD "%{executable_dir_keyword}"
        const char** update_ld_flags(const char *ld_flags[]) {
          const char* lld_bin_dir = getenv("POCL_LINKER_DIR");
          if (!lld_bin_dir) return ld_flags;
          for (int i = 0; ld_flags[i] != NULL; i++) {
              const char* original = ld_flags[i];
              size_t original_len = strlen(original);
              size_t keyword_len = strlen(EXEC_DIR_KEYWORD);
              size_t replacement_len = strlen(lld_bin_dir);
              size_t count = 0;
              for (const char* p = strstr(original, EXEC_DIR_KEYWORD); p; p = strstr(p + keyword_len, EXEC_DIR_KEYWORD)) {
                  count++;
              }
              if (count == 0) continue;
              size_t new_len = original_len + count * (replacement_len - keyword_len);
              char* new_str = (char*)malloc(new_len + 1);
              if (!new_str) continue;
              const char *src = original;
              char *dst = new_str;
              while (*src) {
                  if (strncmp(src, EXEC_DIR_KEYWORD, keyword_len) == 0) {
                      strcpy(dst, lld_bin_dir);
                      dst += replacement_len;
                      src += keyword_len;
                  } else {
                      *dst++ = *src++;
                  }
              }
              *dst = '\0';
              ld_flags[i] = new_str;
          }
          return ld_flags;
        }
        EOF

        # LLC_TRIPLE names the triplet the kernels are built and linked for, and
        # SetupLLVMHostCPU already treats it as overridable -- but this earlier
        # set() overwrites whatever was passed with LLVM's own target triplet,
        # which our cross-built LLVM reports without a version. Clang reads a
        # versionless darwin triplet as a macOS old enough to want -lgcc_s.1 and
        # no -platform_version, so ld64.lld rejects every kernel and the views
        # stay black. Honour the value when one is given.
        grep -qF 'set(LLC_TRIPLE ${LLVM_TARGET_TRIPLE})' cmake/SetupLLVMviaCMake.cmake || { echo "LLC_TRIPLE assignment not found"; exit 1; }
        sed -i 's|^set(LLC_TRIPLE ${LLVM_TARGET_TRIPLE})$|if(NOT LLC_TRIPLE)\n  set(LLC_TRIPLE ${LLVM_TARGET_TRIPLE})\nendif()|' cmake/SetupLLVMviaCMake.cmake

        %{cmake} %{cmake-local-distro} %{cmake-local-darwin} %{cmake-local-darwin-aarch64} %{cmake-local-darwin-x86_64}
  - target_triple == "x86_64-w64-mingw32":
      configure-commands:
      - |
        test -f "%{bindir}/ld.exe" || exit 1 # Check that ld.exe is available
        rm -rf cmake/FindHwloc.cmake
        sed 's/ -fPIC / /g' -i CMakeLists.txt
        sed 's/set(BUILD_SHARED_LIBS_DEFAULT OFF)/set(BUILD_SHARED_LIBS_DEFAULT ON)/' -i CMakeLists.txt
        %{cmake} %{cmake-local-distro} %{cmake-local-win}

build-depends:
- sdk.bst
- clang_host.bst

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
        # For OpenCL-Headers.pc. pocl looks the ICD loader up through
        # pkg-config, and 'ocl-icd' declares OpenCL-Headers in its Requires:
        # without that file the whole lookup fails, pocl reports "Requested
        # build with icd, but ICD loader not found", quietly builds a
        # standalone libOpenCL instead of libpocl.so, and no CPU device is ever
        # registered.
        - opencl-headers.bst
  - target_triple == "x86_64-w64-mingw32":
      depends:
        (>):
        - clang_target.bst
        - binutils.bst
        - opencl-icd-loader.bst
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      depends:
        (>):
        - clang_target.bst
