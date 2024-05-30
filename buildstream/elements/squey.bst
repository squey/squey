kind: cmake

depends:
- base.bst
- freedesktop-sdk.bst:bootstrap/gcc.bst
- freedesktop-sdk.bst:bootstrap/zlib.bst
- freedesktop-sdk.bst:bootstrap/glibc.bst
- freedesktop-sdk.bst:components/openssl.bst
- freedesktop-sdk.bst:components/zip.bst
- freedesktop-sdk.bst:components/unzip.bst
- freedesktop-sdk.bst:bootstrap/pcre2.bst
- freedesktop-sdk.bst:components/ccache.bst
- freedesktop-sdk.bst:components/curl.bst
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

- pigz.bst
- lbzip2.bst
- gperftools.bst
- libhwloc.bst
- pwgen.bst
- mm-common.bst
- libsigcplusplus.bst
- rapidjson.bst
- libnuma.bst
- google-breakpad.bst
- portablecl.bst
- opencl-headers.bst
- libpcap.bst
- lightpcapng.bst
- tshark.bst
- tbb.bst
- boost.bst
- qt6.bst
- kitemmodels.bst
- ksyntaxhighlighting.bst
- pybind11.bst
- python3-pip.bst
- numpy.bst
- libomp.bst
- lcov.bst
- lcov-cobertura.bst
- apache-arrow.bst

build-depends:
- freedesktop-sdk.bst:components/git.bst
- freedesktop-sdk.bst:components/flatpak-xdg-utils.bst 
#- clang-tools-extra.bst
- fix_gdb.bst
- recc.bst
- openssh-portable.bst
- procps.bst
- waypipe.bst
- llvm.bst
- mold.bst

sources:
- kind: local
  path: .

environment:
  LC_ALL: "C.UTF-8"
  RECC_PROJECT_ROOT: ".."
  RECC_CACHE_ONLY: 1
  RECC_CACHE_UPLOAD_LOCAL_BUILD: 1
  RECC_CAS_SERVER: unix:/tmp/casd.sock
  RECC_MAX_THREADS: -1
  
variables:
  build-dir: "build"
  source-dir: "src"
  cmake-local: |
    -DCMAKE_INSTALL_PREFIX=%{prefix} \
    -DARCH_TYPE=corei7 \
    -DUSER_TARGET=%{user_target} \
    -DCMAKE_CXX_COMPILER=%{cxx_compiler} \
    -DCODE_COVERAGE=%{code_coverage}
  copy-launchers: "cp -r buildstream/files/{run_cmd.sh,squey_sh,squey_launcher.sh} %{install-root}%{prefix}/bin"
    
config:
  configure-commands:
    - |
      WARNINGS_AS_ERRORS=$([ "%{cxx_compiler}" = "clang++" ] && echo 1 || echo 0)
      CMAKE_BUILD_TYPE=$([ "%{code_coverage}" = "1" ] && echo "Debug" || echo "RelWithDebInfo")
      cmake -H"%{source-dir}" -B"%{build-dir}" -G"Ninja" %{cmake-local} -DCMAKE_CXX_COMPILER_LAUNCHER="recc" \
                                                          -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
                                                          -DCMAKE_COMPILE_WARNING_AS_ERROR=$WARNINGS_AS_ERRORS
    
  (?):
  - disable_testsuite == False:
      build-commands:
      - |
        export PYTHONPATH=$(bash -c "echo /app/lib/python*/site-packages")
        cd "%{build-dir}" && (ninja && ninja squey_testsuite) || exit 1
        if [ "%{code_coverage}" = "1" ]; then ninja pvcop_testsuite; fi
        TESTS="-R SQUEY_TEST"
        if [ "%{code_coverage}" = "1" ]; then CODE_COVERAGE_COMMAND="-T coverage"; TESTS="-R 'SQUEY_TEST|PVCOP_TEST'"; fi
        mkdir -p /tmp/.squey/squey/tmp-squey/tomjon && cp ../src/pvconfig.ini /tmp/.squey/squey/config.ini && sed 's|\(nraw_tmp=\)/srv/tmp-squey|\1/tmp/.squey/squey/tmp-squey|' -i /tmp/.squey/squey/config.ini
        ../buildstream/files/run_cmd.sh ctest -j $(nproc) --output-junit junit.xml --output-on-failure -T test ${CODE_COVERAGE_COMMAND} ${TESTS} || if [ "%{code_coverage}" = "0" ]; then exit 1; fi
        # Generate code coverage report
        if [ "%{code_coverage}" = "1" ]; then
            ../scripts/gen_code_coverage_report.sh
        fi || exit 1 # fail the testsuite on errors
      install-commands:
      - |
        %{make-install}
        %{copy-launchers}
        if [ "%{keep_build_dir}" = "1" ]; then
          cp --preserve -r "%{build-dir}" "%{install-root}/compilation_build"
        fi
        cp -r %{build-dir}/junit.xml %{build-dir}/code_coverage_report %{install-root} || true
  - disable_testsuite == True:
      build-commands:
        - cd "%{build-dir}" && ninja
      install-commands:
        - "%{make-install}"
        - "%{copy-launchers}"
        
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
