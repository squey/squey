kind: cmake

(@): 
  - buildstream/elements/mysql.yml
  - buildstream/elements/base/target.bst

variables:
  make-args: ""
  cmake-local-extra: >-
    -DWITH_READLINE=OFF
    -DFORCE_UNSUPPORTED_COMPILER=1
    -DWITH_EDITLINE=bundled
    -DWITH_ICU=system
    -DWITH_PROTOBUF=bundled
    -DWITH_CURL=system
    -DWITH_LZ4=system
    -DWITH_ZLIB=system
    -DWITH_ZSTD=system
    -DWITH_SSL=%{prefix}
    -DZLIB_INCLUDE_DIR="%{includedir}"
    -DWITH_SYSTEM_LIBS=ON
    -DCMAKE_CROSSCOMPILING=1
    -DDEFAULT_CHARSET=utf8
    -DDEFAULT_COLLATION=utf8_general_ci
    -DHAVE_C_FLOATING_POINT_FUSED_MADD_EXITCODE=0
    -DHAVE_CXX_FLOATING_POINT_FUSED_MADD_EXITCODE=0

environment:
  CXXFLAGS: "-fpermissive"

config:
  configure-commands:
  - |
    sed 's|/usr/bin/libtool|%{toolchain-dir}/%{host}-libtool|g' -i cmake/os/Darwin.cmake cmake/libutils.cmake
    #sed '410,424d' -i configure.cmake
    
    sed '115d' -i utilities/CMakeLists.txt
    sed 's/BaseTsd.h/basetsd.h/g' -i include/my_inttypes.h
    rm -rf extra/zlib
    %{cmake}
    
build-depends:
- sdk.bst
- icu.bst
- curl.bst
- openssl.bst
- zlib.bst
- zstd.bst
- lz4.bst
- abseil-cpp.bst
- mysql_host.bst

(?):
- (target_triple == "x86_64-w64-mingw32"):
    build-depends:
      (>):
      - zlib.bst
      - base/winsdk-10.bst