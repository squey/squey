(?):
    - (target_triple == "x86_64-apple-darwin"):
        variables:
            target-arch: "x86_64"
            macho-arch: "%{target-arch}"
            cc-compiler: "o64-clang"
            cxx-compiler: "o64-clang++"
            openssl-target: "darwin64-%{target-arch}-cc"
    - (target_triple == "aarch64-apple-darwin"):
        variables:
            target-arch: "aarch64"
            macho-arch: "arm64"
            cc-compiler: "oa64-clang"
            cxx-compiler: "oa64-clang++"
            openssl-target: "darwin64-arm64-cc"
    - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
        variables:
            host-prefix: "/app"
            prefix: "/mac"
            macos-version: "24.1"
            macos-sdk-version: "15.1"
            deployment-target-version: "10.15"
            target-vendor: "apple"
            target-platform: "darwin"
            host: "%{target-arch}-%{target-vendor}-%{target-platform}%{macos-version}"
            osxcross-dir: "%{host-prefix}/osxcross"
            toolchain-dir: "%{osxcross-dir}/bin"
            sdk-dir: "%{osxcross-dir}/SDK/MacOSX%{macos-sdk-version}.sdk"
            dylib-extension: "dylib"
            exe-extension: ""
            cmake-global: >-
                -DCMAKE_TOOLCHAIN_FILE="%{osxcross-dir}/toolchain.cmake"
                -DCMAKE_FIND_ROOT_PATH="%{prefix}"
                -DCMAKE_OSX_DEPLOYMENT_TARGET="%{deployment-target-version}"
            cmake-local-extra: ""
            conf-global: |
                CC=%{cc-compiler} CXX=%{cxx-compiler} RANLIB="%{host}-ranlib" AR="%{host}-ar" --host=%{host}
            make-args: >-
                UNAME="Darwin"
                CC="%{cc-compiler}"
                CXX="%{cxx-compiler}"
                AR="%{host}-ar"
                RANLIB="%{host}-ranlib"
                PREFIX="%{prefix}"
            qt-crossplatform-options: >-
                -DCMAKE_CXX_COMPILER_TARGET="%{target-arch}-apple-darwin"
                -DQT_INTERNAL_APPLE_SDK_VERSION="%{macos-sdk-version}"
                -DQT_QMAKE_TARGET_MKSPEC=macx-clang
                -DCMAKE_SYSROOT="%{sdk-dir}"
                -DCMAKE_OSX_ARCHITECTURES="%{macho-arch}"
                -DINPUT_linker="%{toolchain-dir}/%{host}-ld"
                -DQT_HOST_PATH="%{host-prefix}"
            boost-toolset: "clang-osxcross"
            boost-target-os: "%{target-platform}"
            boost-cxx-flags: "--target=%{target-arch}-apple-darwin"
            boost-link-flags: "-fuse-ld=%{toolchain-dir}/%{host}-ld"
            openssl-arch-conf: enable-ec_nistp_64_gcc_128
            icu-platform: "MacOSX"
        environment:
            PATH: "%{toolchain-dir}:%{host-prefix}/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            LD_LIBRARY_PATH: "%{osxcross-dir}/lib:%{host-prefix}/lib/clang/18/lib/darwin/:/usr/lib:%{host-prefix}/lib"
            OSXCROSS_TARGET_DIR: "%{osxcross-dir}"
            OSXCROSS_SDK: "%{sdk-dir}"
            OSXCROSS_PKG_CONFIG_USE_NATIVE_VARIABLES: 1
            OSXCROSS_HOST: "%{host}"
            OSXCROSS_TARGET: "%{host}"
            MACOSX_DEPLOYMENT_TARGET: "%{deployment-target-version}"
    - (target_triple == "x86_64-w64-mingw32"):
        variables:
            host-prefix: "/app"
            prefix: "/win"
            host: "%{target_triple}"
            target-arch: "x86_64"
            dylib-extension: "dll"
            exe-extension: ".exe"
            target-platform: "windows"
            sdk-dir: "/usr/%{host}"
            toolchain-dir: "/usr/bin"
            cc-compiler: "%{host}-gcc"
            cxx-compiler: "%{host}-g++"
            cmake-local-extra: ""
            deployment-target-version: ""
            conf-global: |
                CC="%{cc-compiler}" CXX="%{cxx-compiler}" AR="%{host}-ar" RANLIB="%{host}-ranlib" --host="%{host}"
            cmake-global: >-
                -DCMAKE_CROSSCOMPILING=TRUE
                -DCMAKE_SYSTEM_PROCESSOR="x86_64"
                -DCMAKE_SYSTEM_NAME="Windows"
                -DCMAKE_C_COMPILER="%{cc-compiler}"
                -DCMAKE_CXX_COMPILER="%{cxx-compiler}"
                -DCMAKE_FIND_ROOT_PATH="%{prefix}"
                -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
                -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
                -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
                -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
                -DCMAKE_AR="%{toolchain-dir}/%{host}-ar"
                -DCMAKE_RANLIB="%{toolchain-dir}/%{host}-ranlib"
            boost-toolset: "gcc-mingw"
            boost-target-os: "%{target-platform} architecture=x86 address-model=64 threadapi=win32 link=shared"
            boost-cxx-flags: ""
            boost-link-flags: ""
            openssl-target: "mingw64"
            openssl-arch-conf: ""
            icu-platform: "MinGW"
            qt-crossplatform-options: >-
                -DQT_QMAKE_TARGET_MKSPEC="win32-g++"
                -DINPUT_linker="%{toolchain-dir}/%{host}-ld"
                -DQT_HOST_PATH="%{host-prefix}"
        environment:
            PATH: "%{toolchain-dir}:%{host-prefix}/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            LD_LIBRARY_PATH: "%{libdir}:/usr/%{host}/lib"
            C_INCLUDE_PATH: "%{includedir}"
            CPLUS_INCLUDE_PATH: "%{includedir}"
            PREFIX: "%{toolchain-dir}/%{host}-"
    - (target_triple == "x86_64-linux-gnu"):
        variables:
            host-prefix: "%{prefix}"
            host: "%{build-triplet}"
            target-arch: "x86_64"
            dylib-extension: "so"
            exe-extension: ""
            deployment-target-version: ""
            target-platform: "linux"
            cc-compiler: gcc
            cxx-compiler: "/usr/bin/%{cxx_compiler}"
            cmake-local-extra: ""
            toolchain-dir: "/usr/bin"
            sdk-dir: ""
            boost-toolset: gcc
            boost-target-os: "%{target-platform}"
            boost-cxx-flags: ""
            boost-link-flags: ""