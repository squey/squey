kind: cmake
description: |

    Crashpad is a crash-reporting system: an out-of-process handler captures a
    minidump, stores it in a local database and uploads it after consent.

    The getsentry fork is used rather than upstream because it ships a CMake
    build for all three targets (upstream only builds with GN/depot_tools) and
    it carries the mingw-w64 support upstream never merged. Its StartHandler()
    also takes a 'crash_reporter' executable, which keeps the crash reporter
    dialog of Squey on the crash path.

sources:
- kind: git
  url: github:getsentry/crashpad.git
  ref: aae505d3daf73e8a48136ccc7398663f16096712
  checkout-submodules: True

- kind: patch
  path: buildstream/patches/crashpad/mingw-offsetof-non-constant.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/crashpad/config-cmake-target-order.patch
  strip-level: 1

- kind: patch
  path: buildstream/patches/crashpad/keep-report-for-crash-reporter.patch
  strip-level: 1

(@): buildstream/elements/base/target.yml

variables:
  cmake-local: >-
    -DCRASHPAD_ZLIB_SYSTEM=ON
    -DCRASHPAD_ENABLE_INSTALL=ON
    -DCRASHPAD_ENABLE_INSTALL_DEV=ON
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    %{cmake-crashpad-extra}

  # No extra argument is needed outside of Windows.
  cmake-crashpad-extra: ""

config:
  configure-commands:
  - "%{cmake}"

depends:
- zlib.bst

build-depends:
- sdk.bst

(?):
  - (target_triple == "x86_64-w64-mingw32"):
      variables:
        # enable_language(ASM_MASM) runs before the CMakeLists looks for jwasm,
        # so CMAKE_ASM_MASM_COMPILER is already set to 'ml' by then and the
        # auto-detection is skipped: point at jwasm explicitly. The flags are
        # left unset on purpose, the CMakeLists then defaults them to
        # "-win64 -10".
        cmake-crashpad-extra: >-
          -DCMAKE_ASM_MASM_COMPILER=%{host-prefix}/bin/jwasm
          -DCMAKE_CXX_FLAGS="-isystem %{build-root}/mingw_compat_include"
          -DCMAKE_C_FLAGS="-isystem %{build-root}/mingw_compat_include"
      config:
        configure-commands:
          (<):
          # mingw-w64 ships these headers in lower case only, while
          # mini_chromium and the Crashpad WER module include them with the
          # casing of the Windows SDK.
          - |
            mkdir -p "%{build-root}/mingw_compat_include"
            echo "#include <ntsecapi.h>" > "%{build-root}/mingw_compat_include/NTSecAPI.h"
            echo "#include <windows.h>" > "%{build-root}/mingw_compat_include/Windows.h"
      build-depends:
        (>):
        - jwasm.bst
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      # util/CMakeLists.txt shells out to mach/mig.py, which resolves the SDK
      # through the xcrun of osxcross and then invokes mig from the PATH.
      config:
        install-commands:
          (>):
          # The handler links against libz, whose install name is
          # @rpath/libz.1.dylib, but CMake leaves it without any LC_RPATH. In the
          # bundle it sits in Contents/MacOS while the dylibs are in
          # Contents/Frameworks, same as the squey executable.
          - |
            %{toolchain-dir}/%{host}-install_name_tool -add_rpath @executable_path/../Frameworks \
              "%{install-root}%{bindir}/crashpad_handler"
      build-depends:
        (>):
        - mig.bst
