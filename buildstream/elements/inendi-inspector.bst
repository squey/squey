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
- libpcap.bst
- lightpcapng.bst
- tshark.bst
- tbb.bst
- boost.bst
- qt6.bst
- open62541.bst
- kitemmodels.bst
- ksyntaxhighlighting.bst
# - erf.bst
- qtopcua.bst
- pybind11.bst
- python3-pip.bst
# - conda.bst
- numpy.bst
- inendi-inspector-doc.bst
- libomp.bst
- lcov.bst
- lcov-cobertura.bst

build-depends:
- freedesktop-sdk.bst:components/git.bst

sources:
- kind: local
  path: .

environment:
  CCACHE_DIR: ".ccache"
  
variables:
  build-dir: "build"
  cmake-local: |
    -DCMAKE_INSTALL_PREFIX=%{prefix} \
    -DARCH_TYPE=corei7 \
    -DUSER_TARGET=%{user_target} \
    -DCMAKE_CXX_COMPILER=%{cxx_compiler} \
    -DCODE_COVERAGE=%{code_coverage}
  copy-launchers: "cp -r buildstream/files/{run_cmd.sh,inspector_launcher.sh} %{install-root}%{prefix}/bin"
    
config:
  configure-commands:
    - |
      CCACHE_COMPILER=$([ "%{code_coverage}" = "1" ] && echo "%{cxx_compiler}" || echo "ccache")
      [ "%{code_coverage}" = "1" ] && rm -rf "%{build-dir}" || true
      if grep "CMAKE_CXX_COMPILER:STRING" "%{build-dir}/CMakeCache.txt" | grep -v -q "%{cxx_compiler}" ; then rm -rf "%{build-dir}/CMakeCache.txt"; fi
      WARNINGS_AS_ERRORS=$([ "%{cxx_compiler}" = "clang++" ] && echo 1 || echo 0)
      CMAKE_BUILD_TYPE=$([ "%{code_coverage}" = "1" ] && echo "Debug" || echo "RelWithDebInfo")
      cmake -B"%{build-dir}" -H. -G"Ninja" %{cmake-local} -DCMAKE_CXX_COMPILER_LAUNCHER=$CCACHE_COMPILER \
                                                          -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
                                                          -DWARNINGS_AS_ERRORS=$WARNINGS_AS_ERRORS
    
  (?):
  - disable_testsuite == False:
      build-commands:
        - cd "%{build-dir}" && ninja && ninja inspector_testsuite
        - cd "%{build-dir}" ; if [ "%{code_coverage}" = "1" ]; then ninja pvcop_testsuite; fi
        - cp -r scripts "%{build-dir}"
      install-commands:
        - "%{make-install}"
        - cp --preserve -r . "%{install-root}/compilation"
        - "%{copy-launchers}"
  - disable_testsuite == True:
      build-commands:
        - "%{make}"
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
