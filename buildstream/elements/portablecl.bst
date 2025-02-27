kind: cmake
description: |

    PoCL is a portable open source (MIT-licensed) implementation of the OpenCL standard.

sources:
- kind: git
  url: github:pocl/pocl
  ref: a2e0199f794c8dfbcd92b775cd4f82ba05e816a9

(@): buildstream/elements/base/target.bst

environment:
  PKG_CONFIG_PATH: "%{bindir}/pkgconfig"
  VERBOSE: 1

variables:
  generator: "Unix Makefiles"
  cmake-local-distro: ''
  cmake-local-linux: ''
  cmake-local-darwin-aarch64: ''
  cmake-local: >-
    -DINSTALL_OPENCL_HEADERS=1
    -DCMAKE_BUILD_TYPE=Release
    -DENABLE_TBB_DEVICE=1
    -DCMAKE_FIND_DEBUG_MODE=OFF

  executable_dir_keyword: "@executable_dir"
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
    -DSTATIC_LLVM=OFF
    -DDL_H="%{sdk-dir}/usr/include/dlfcn.h"

  cmake-local-win: >-
    -DCLANG="/app/bin/clang"
    -DCLANGXX="/app/bin/clang++"
    -DCMAKE_C_FLAGS="-L%{libdir} -isystem %{includedir}"
    -DCMAKE_CXX_FLAGS="-L%{libdir} -isystem %{includedir}"
    -DBUILD_PREFIX="%{host-prefix}"
    -DHOST_PREFIX="%{prefix}"
    -DLLVM_HOST_TARGET="%{host}"
    -DCMAKE_SYSROOT="%{prefix}"
    -DLLVM_BINDIR="%{host-prefix}/bin"
    -DLLVM_LIBDIR="%{libdir}"
    -DSTATIC_LLVM=TRUE
    -DLLVM_HOST_TRIPLE="x86_64-pc-win32-gnu"
    -DCMAKE_SIZEOF_VOID_P=8
    -DLLVM_PREFIX="%{prefix}"
    -DLLVM_VERSION_FULL=18.1.8
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
    -DDEFAULT_ENABLE_ICD=OFF

  (?):
  - (target_triple == "x86_64-linux-gnu" or target_triple == "x86_64-apple-darwin" or target_triple == "x86_64-w64-mingw32"):
      cmake-local-distro: >-
        -DKERNELLIB_HOST_CPU_VARIANTS=distro
  - target_triple == "aarch64-apple-darwin":
      cmake-local-darwin-aarch64: >-
        -DLLC_TRIPLE="%{host}"
        -DLLC_HOST_CPU="apple-m1"
        -DHOST_DEVICE_BUILD_HASH="%{macho-arch}-apple-darwin%{deployment-target-version}.0"
  - target_triple == "x86_64-linux-gnu":
      cmake-local-linux: >-
        -DEXTRA_HOST_LD_FLAGS="-L%{libdir}"

config:
  (?):
  - target_triple == "x86_64-linux-gnu":
      configure-commands:
      - |
        %{cmake} %{cmake-local-distro} %{cmake-local-linux}
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
        %{cmake} %{cmake-local-distro} %{cmake-local-darwin} %{cmake-local-darwin-aarch64}
  - target_triple == "x86_64-w64-mingw32":
      configure-commands:
      - |
        test -f "%{bindir}/ld.exe" || exit 1 # Check that ld.exe is available
        rm -rf cmake/FindHwloc.cmake
        sed 's/ -fPIC / /g' -i CMakeLists.txt
        sed 's/@POCL_INSTALL_PRIVATE_DATADIR@/pocl/' -i config.h.in.cmake
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
  - target_triple == "x86_64-w64-mingw32":
      depends:
        (>):
        - clang_target.bst
        - binutils.bst
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      depends:
        (>):
        - clang_target.bst
#  - target_triple == "x86_64-w64-mingw32":
#      depends:
#        (>):
#        - icd-loader.bst

# -DLLVM_LIBS="-lLLVMWindowsManifest -lLLVMXRay -lLLVMLibDriver -lLLVMDlltoolDriver -lLLVMTextAPIBinaryReader -lLLVMCoverage -lLLVMLineEditor
#    -lLLVMX86TargetMCA -lLLVMX86Disassembler -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMX86Desc -lLLVMX86Info -lLLVMOrcDebugging -lLLVMOrcJIT -lLLVMWindowsDriver
#    -lLLVMMCJIT -lLLVMJITLink -lLLVMInterpreter -lLLVMExecutionEngine -lLLVMRuntimeDyld -lLLVMOrcTargetProcess -lLLVMOrcShared -lLLVMDWP -lLLVMDebugInfoLogicalView -lLLVMDebugInfoGSYM -lLLVMOption
#    -lLLVMObjectYAML -lLLVMObjCopy -lLLVMMCA -lLLVMMCDisassembler -lLLVMLTO -lLLVMPasses -lLLVMHipStdPar -lLLVMCFGuard -lLLVMCoroutines -lLLVMipo -lLLVMVectorize -lLLVMLinker -lLLVMInstrumentation
#    -lLLVMFrontendOpenMP -lLLVMFrontendOffloading -lLLVMFrontendOpenACC -lLLVMFrontendHLSL -lLLVMFrontendDriver -lLLVMExtensions -lLLVMDWARFLinkerParallel -lLLVMDWARFLinkerClassic -lLLVMDWARFLinker
#    -lLLVMGlobalISel -lLLVMMIRParser -lLLVMAsmPrinter -lLLVMSelectionDAG -lLLVMCodeGen -lLLVMTarget -lLLVMObjCARCOpts -lLLVMCodeGenTypes -lLLVMIRPrinter -lLLVMInterfaceStub -lLLVMFileCheck
#    -lLLVMFuzzMutate -lLLVMScalarOpts -lLLVMInstCombine -lLLVMAggressiveInstCombine -lLLVMTransformUtils -lLLVMBitWriter -lLLVMAnalysis -lLLVMProfileData -lLLVMSymbolize -lLLVMDebugInfoBTF
#    -lLLVMDebugInfoPDB -lLLVMDebugInfoMSF -lLLVMDebugInfoDWARF -lLLVMObject -lLLVMTextAPI -lLLVMMCParser -lLLVMIRReader -lLLVMAsmParser -lLLVMMC -lLLVMDebugInfoCodeView -lLLVMBitReader
#    -lLLVMFuzzerCLI -lLLVMCore -lLLVMRemarks -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMTargetParser -lLLVMTableGen -lLLVMSupport -lLLVMDemangle"