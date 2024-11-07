kind: cmake

(@): 
  - buildstream/elements/mysql.yml
  - buildstream/elements/base/target.bst

variables:
  make-args: ""
  cmake-local-extra: >-
    -DCURSES_INCLUDE_PATH="%{prefix}/include/ncurses"
    -DFORCE_UNSUPPORTED_COMPILER=1
    -DWITH_EDITLINE=bundled
    -DWITH_ICU=system
    -DWITH_PROTOBUF=bundled
    -DWITH_CURL=system
    -DWITH_LZ4=system
    -DWITH_ZLIB=system
    -DWITH_ZSTD=system
    -DWITH_SSL=system
    -DZLIB_INCLUDE_DIR="%{prefix}/include"
    -DWITH_SYSTEM_LIBS=ON
    -DCMAKE_SYSTEM_NAME="Linux"
    -DCMAKE_CROSSCOMPILING=1
    -DDEFAULT_CHARSET=utf8
    -DDEFAULT_COLLATION=utf8_general_ci

config:
  configure-commands:
  - |
    sed 's|/usr/bin/libtool|%{toolchain-dir}/%{host}-libtool|g' -i cmake/os/Darwin.cmake cmake/libutils.cmake
    sed '410,424d' -i configure.cmake
    rm -rf extra/zlib
    %{cmake}
    
build-depends:
- sdk.bst
- icu.bst
- curl.bst
- ncurses.bst
- openssl.bst
- zlib.bst
- zstd.bst
- lz4.bst
- abseil-cpp.bst
- mysql_host.bst
