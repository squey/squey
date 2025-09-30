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
  RECC_PROJECT_ROOT: ".."
  RECC_CACHE_ONLY: 1
  RECC_CACHE_UPLOAD_LOCAL_BUILD: 1
  RECC_CAS_SERVER: unix:/tmp/casd.sock
  RECC_MAX_THREADS: -1

variables:
  build-dir: "build/%{target_triple}/%{cxx_compiler}/build"
  source-dir: "."
  cmake-local: >-
    -DUSER_TARGET=%{user_target}
    -DCODE_COVERAGE=%{code_coverage}
    %{cmake-local-extra}
  (?):
  - target_triple == "x86_64-linux-gnu":
      cmake-local-extra: >-
        -DCMAKE_CXX_COMPILER="/usr/bin/%{cxx_compiler}"
  - target_triple != "x86_64-linux-gnu":
      cmake-local-extra: ""

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
      %{cmake} "%{source-dir}" \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=$WARNINGS_AS_ERRORS

  (?):
  - disable_testsuite == False:
      build-commands:
      - |
        export PYTHONPATH=$(bash -c "echo /app/lib/python*/site-packages")
        SOURCE_DIR=$(pwd)
        cd "%{build-dir}" && (ninja && ninja squey_testsuite) || exit 1
        [ "%{target_triple}" != "x86_64-linux-gnu" ] && exit 0
        if [ "%{code_coverage}" = "1" ]; then ninja pvcop_testsuite; fi
        TESTS="-R SQUEY_TEST"
        if [ "%{code_coverage}" = "1" ]; then CODE_COVERAGE_COMMAND="-T coverage"; TESTS="-R 'SQUEY_TEST|PVCOP_TEST'"; fi
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
        - cd "%{build-dir}" && ninja
      install-commands:
      - |
        %{install-all}
        # embed cross-compiled testsuites
        [ "%{target_triple}" != "x86_64-linux-gnu" ] && cmake --install "%{build-dir}" --component tests --prefix "%{install-root}%{prefix}/tests"

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
- sdk.bst
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
        - lcov.bst
        - lcov-cobertura.bst
  - target_triple == "x86_64-linux-gnu":
      build-depends:
        (>):
        #- clang-tools-extra.bst
        - llvm.bst
        - iwyu.bst
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
