kind: cmake

sources:
- kind: local
  path: "src"
- kind: local
  path: "VERSION.txt"
- kind: local
  path: "CHANGELOG"
- kind: local
  path: "LICENSE"
- kind: local
  path: buildstream/dev_shell_bashrc
  directory: buildstream
- kind: local
  path: buildstream/env.conf
  directory: buildstream
- kind: local
  path: buildstream/scripts
  directory: buildstream/scripts
- kind: local
  path: buildstream/sshd
  directory: buildstream/sshd
- kind: local
  path: buildstream/files/flatpak
  directory: buildstream/files/flatpak
- kind: local
  path: buildstream/files/macos_bundle/libSystemStub.tbd

(@): buildstream/elements/base/target.yml

environment:
  LC_ALL: "C.UTF-8"
  # Cache individual compile commands through the REAPI socket exposed by
  # BuildStream in the sandbox (see the 'sandbox' section below). Actions
  # missing from the cache are built locally and their result uploaded.
  RECC_CACHE_ONLY: 1
  RECC_CACHE_UPLOAD_LOCAL_BUILD: 1
  RECC_SERVER: unix:/tmp/reapi.sock
  # Include headers found in absolute paths (/usr/include, /app/include) in the
  # action key, otherwise a dependency update would silently yield stale hits
  RECC_DEPS_GLOBAL_PATHS: 1
  # The sandbox build directory is stable across builds, no need to make the
  # paths relative (which would break the global dependency paths above)
  RECC_NO_PATH_REWRITE: 1
  # recc detects dependencies by appending '-M -v' to the compile command, which
  # makes '-c' unused. Combined with the -Werror of the release build, clang turns
  # that warning into an error and recc silently falls back to uncached local
  # compilation. Only the dependency command is patched here, the actual
  # compilation keeps -Werror untouched. GCC does not need this.
  RECC_DEPS_ENV_CCC_OVERRIDE_OPTIONS: "+-Wno-error=unused-command-line-argument"
  RECC_MAX_THREADS: -1

sandbox:
  remote-apis-socket:
    path: /tmp/reapi.sock
    # Required to let recc upload the results of locally built actions,
    # as it can not go through remote execution in cache-only mode
    action-cache-enable-update: true

variables:
  build-dir: "build/%{target_triple}/%{cxx_compiler}/build"
  source-dir: "."
  quiet-flag: ""
  cmake-local: >-
    -DUSER_TARGET=%{user_target}
    -DCODE_COVERAGE=%{code_coverage}
    %{cmake-local-extra}
  # recc only recognizes plain compiler basenames (cc, c++, gcc, g++, clang, clang++):
  # a cross compilation prefix makes it report "not a compiler command" and silently
  # skip caching. The substitution is done in the compiler launcher rather than in
  # CMAKE_<LANG>_COMPILER, so that CMake still sees the prefixed compiler and keeps
  # deriving the matching binutils from it (windres, dlltool, nm, objcopy...).
  recc-bin: "%{build-root}/recc-bin"
  (?):
  - quiet_compilation == True:
      quiet-flag: "--quiet"
  - target_triple == "x86_64-linux-gnu":
      cmake-local-extra: >-
        -DCMAKE_CXX_COMPILER="/usr/bin/%{cxx_compiler}"
      recc-c-launcher: "recc"
      recc-cxx-launcher: "recc"
  - target_triple != "x86_64-linux-gnu":
      cmake-local-extra: ""
      recc-c-launcher: "%{recc-bin}/launch-cc"
      recc-cxx-launcher: "%{recc-bin}/launch-c++"

  install-linux-launchers: 'bash -c "cp -r buildstream/files/flatpak/{run_cmd.sh,squey_sh,squey_launcher.sh} %{install-root}%{prefix}/bin"'
  install-linux-deps:
    install -D /usr/lib/%{gcc_triplet}/crti.o -t %{install-root}%{libdir} &&
    install -D /usr/lib/%{gcc_triplet}/crtn.o -t %{install-root}%{libdir} &&
    install -D /usr/lib/%{gcc_triplet}/libgcc_s.so -t %{install-root}%{libdir} &&
    install -D /usr/lib/%{gcc_triplet}/libgcc_s.so.1 -t %{install-root}%{libdir} &&
    install -D /usr/lib/%{gcc_triplet}/libc.so -t %{install-root}%{libdir} &&
    sed -i 's|/usr/lib/x86_64-linux-gnu/||g;s|/lib64/||' %{install-root}%{libdir}/libc.so &&
    install -D /usr/lib/%{gcc_triplet}/libc_nonshared.a -t %{install-root}%{libdir} &&
    cp /usr/lib/%{gcc_triplet}/libm.so.6 %{install-root}%{libdir}/libm.so &&
    find /usr/lib/gcc/%{triplet} -name libgcc.a -exec install -D \{\} -t %{install-root}%{libdir} \; &&
    find /usr/lib/gcc/%{triplet} -name crtbeginS.o -exec install -D \{\} -t %{install-root}%{libdir} \; &&
    find /usr/lib/gcc/%{triplet} -name crtendS.o -exec install -D \{\} -t %{install-root}%{libdir} \;
  install-all: |
    %{make-install} &&
    if [ "%{target_triple}" = "x86_64-linux-gnu" ]; then %{install-linux-deps} && %{install-linux-launchers} ; fi &&
    if [ "%{target-platform}" = "darwin" ]; then mkdir -p "%{install-root}%{libdir}" && cp libSystemStub.tbd "%{install-root}%{libdir}/libSystem.tbd" && %{toolchain-dir}/%{host}-install_name_tool -rpath %{libdir} @executable_path/../Frameworks %{install-root}%{bindir}/squey ; fi

config:
  configure-commands:
    - |
      WARNINGS_AS_ERRORS=$([ "%{code_coverage}" = "1" ] && echo 0 || echo 1)
      CMAKE_BUILD_TYPE=$([ "%{code_coverage}" = "1" ] && echo "Debug" || echo "RelWithDebInfo")
      # CMake runs '<launcher> <compiler> <flags>'. The launcher replaces the compiler
      # by a wrapper whose basename recc accepts, and passes the real one along: the
      # wrapper execs it, so it keeps its own argv[0]. The osxcross drivers need that,
      # as they are all symlinks to a single wrapper binary that parses argv[0] to
      # know its target. See the 'recc-bin' variable.
      if [ "%{target_triple}" != "x86_64-linux-gnu" ]; then
        mkdir -p "%{recc-bin}"
        for lang in cc c++; do
          printf '#!/bin/sh\nexec "$SQUEY_REAL_COMPILER" "$@"\n' > "%{recc-bin}/$lang"
          printf '#!/bin/sh\nSQUEY_REAL_COMPILER="$1"; export SQUEY_REAL_COMPILER; shift\nexec recc "%{recc-bin}/%s" "$@"\n' "$lang" > "%{recc-bin}/launch-$lang"
          chmod +x "%{recc-bin}/$lang" "%{recc-bin}/launch-$lang"
        done
      fi
      %{cmake} "%{source-dir}" \
        -DCMAKE_C_COMPILER_LAUNCHER="%{recc-c-launcher}" \
        -DCMAKE_CXX_COMPILER_LAUNCHER="%{recc-cxx-launcher}" \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=$WARNINGS_AS_ERRORS

  (?):
  - disable_testsuite == False:
      build-commands:
      - |
        export PYTHONPATH=$(bash -c "echo /app/lib/python*/site-packages")
        SOURCE_DIR=$(pwd)
        cd "%{build-dir}" && (ninja %{quiet-flag} && ninja %{quiet-flag} squey_testsuite) || exit 1
        [ "%{target_triple}" != "x86_64-linux-gnu" ] && exit 0
        if [ "%{code_coverage}" = "1" ]; then ninja %{quiet-flag} pvcop_testsuite; fi
        TESTS="-R SQUEY_TEST"
        # No quotes around the regex: the expansion of ${TESTS} below is not
        # subject to quote removal, so ctest would receive them as part of it
        if [ "%{code_coverage}" = "1" ]; then CODE_COVERAGE_COMMAND="-T coverage"; TESTS="-R SQUEY_TEST|PVCOP_TEST"; fi
        mkdir -p /tmp/.squey/squey/tmp-squey/tomjon && cp $SOURCE_DIR/pvconfig.ini /tmp/.squey/squey/config.ini && sed 's|\(nraw_tmp=\)/srv/tmp-squey|\1/tmp/.squey/squey/tmp-squey|' -i /tmp/.squey/squey/config.ini
        $SOURCE_DIR/buildstream/files/flatpak/run_cmd.sh ctest -j $(nproc) --output-junit junit.xml --output-on-failure -T test ${CODE_COVERAGE_COMMAND} ${TESTS} || if [ "%{code_coverage}" = "0" ]; then exit 1; fi
        # Generate code coverage report
        if [ "%{code_coverage}" = "1" ]; then
            $SOURCE_DIR/scripts/gen_code_coverage_report.sh
        fi || exit 1 # fail the testsuite on errors
      install-commands:
      - |
        %{install-all}
        if [ "%{keep_build_dir}" = "1" ]; then
          cp --preserve -r "%{build-dir}" "%{install-root}/compilation_build"
        fi
        cp -r %{build-dir}/junit.xml %{build-dir}/code_coverage_report %{install-root} || true
  - disable_testsuite == True:
      build-commands:
        - cd "%{build-dir}" && ninja %{quiet-flag}
      install-commands:
      # No testsuite to embed here: the test executables are EXCLUDE_FROM_ALL and this
      # branch only builds the default target. Cross-compiled testsuites are packaged
      # from the 'disable_testsuite == False' branch, whose regular install already
      # lays them out under %{prefix}/tests, which is what testsuite.zip is built from.
      - |
        %{install-all}

  strip-commands:
    - |
      cd "/buildstream-install%{prefix}" && find -type f \
      '(' -perm -111 -o -name '*.so*' \
          -o -name '*.cmxs' -o -name '*.node' ')' \
      -exec sh -ec \
      'read -n4 hdr <"$1" # check for elf header
       if [ "$hdr" != "$(printf \\x7fELF)" ]; then
           exit 0
       fi
       debugfile="/buildstream-install/app/lib/debug/$1"
       mkdir -p "$(dirname "$debugfile")"
       objcopy --only-keep-debug --compress-debug-sections "$1" "$debugfile"
       chmod 644 "$debugfile"
       strip --remove-section=.comment --remove-section=.note --strip-unneeded "$1"
       objcopy --add-gnu-debuglink "$debugfile" "$1"' - {} ';'

build-depends:
# The digest of the toolchain and of the base environment is part of the recc
# action key: recc hashes neither the compiler nor the libraries it links against
- filename: sdk.bst
  config:
    digest-environment: RECC_REMOTE_PLATFORM_chrootRootDigest
- freedesktop-sdk.bst:bootstrap/gcc.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:bootstrap/glibc.bst
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:components/zip.bst
- freedesktop-sdk.bst:components/unzip.bst
- freedesktop-sdk.bst:bootstrap/pcre2.bst
- freedesktop-sdk.bst:components/ccache.bst
- freedesktop-sdk.bst:components/icu.bst
- freedesktop-sdk.bst:components/gdb.bst
- freedesktop-sdk.bst:components/libarchive.bst
- freedesktop-sdk.bst:components/opencl.bst
- freedesktop-sdk.bst:components/libglvnd.bst
- freedesktop-sdk.bst:components/mesa-headers.bst
- freedesktop-sdk.bst:components/strace.bst
- freedesktop-sdk.bst:components/nsswitch-config.bst
- freedesktop-sdk.bst:components/ca-certificates.bst
- freedesktop-sdk.bst:components/dejavu-fonts.bst
- freedesktop-sdk.bst:components/gnu-free-fonts.bst
- freedesktop-sdk.bst:components/liberation-fonts.bst
- freedesktop-sdk.bst:components/xorg-lib-x11.bst
- freedesktop-sdk.bst:components/xorg-lib-xcb.bst
- freedesktop-sdk.bst:components/xorg-proto-xcb.bst
- freedesktop-sdk.bst:components/xcb-util.bst
- freedesktop-sdk.bst:components/xcb-util-cursor.bst
- freedesktop-sdk.bst:components/xcb-util-wm.bst
- freedesktop-sdk.bst:components/xcb-util-renderutil.bst
- freedesktop-sdk.bst:components/xcb-util-image.bst
- freedesktop-sdk.bst:components/xcb-util-keysyms.bst
- freedesktop-sdk.bst:components/xorg-lib-xcursor.bst
- freedesktop-sdk.bst:components/python3-pip.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/git.bst
- freedesktop-sdk.bst:components/flatpak-xdg-utils.bst
- freedesktop-sdk.bst:components/buildbox.bst
- target_env_vars.bst
- openssh-portable.bst
- procps.bst
- waypipe.bst
- mold.bst

depends:
- libhwloc.bst
- pwgen.bst
- mm-common.bst
- libsigcplusplus.bst
- rapidjson.bst
- google-breakpad.bst ###
- portablecl.bst
- opencl-headers.bst
- libpcap.bst
- lightpcapng.bst
- tbb.bst
- boost.bst
- pybind11.bst
- kitemmodels.bst
- ksyntaxhighlighting.bst
- apache-arrow.bst
- simde.bst
- 7zip_host.bst

(?):
  - target_triple == "x86_64-linux-gnu":
      depends:
        (>):
        - qt6_host.bst
        - tshark.bst
        - libnuma.bst
        - gperftools.bst
        - numpy.bst
        - python3-pip.bst
  - target_triple == "x86_64-linux-gnu":
      build-depends:
        (>):
        #- clang-tools-extra.bst
        - llvm.bst
        - iwyu.bst
        - gcovr.bst
  - target_triple != "x86_64-linux-gnu":
      depends:
        (>):
        - qt6_target.bst
        - tshark_bin.bst
        - python3_bin.bst
        - numpy_bin.bst
        - icu.bst
        - libarchive.bst
        - curl.bst
        - xz.bst
        - zstd.bst
        - 7zip_target.bst
  - target_triple != "x86_64-linux-gnu":
      build-depends:
        (>):
        - qt6_host.bst
  - target_triple != "x86_64-w64-mingw32":
      depends:
        (>):
        - lbzip2.bst
        - pigz.bst
        - libomp.bst
  - target_triple == "x86_64-w64-mingw32":
      depends:
        (>):
        - 7zip_target.bst
        - zip.bst
        - pigz_bin.bst
        - pbzip2.bst
        - python3-pip_bin.bst
