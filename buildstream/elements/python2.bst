kind: autotools
description: Python 2
  
environment:
  CC: "gcc"
  CXX: "g++"

variables:
  indep-libdir: "%{prefix}/lib"
  conf-local: |
    --prefix=/usr \
    --libdir=/usr/lib/x86_64-linux-gnu \
    --enable-shared \
    --with-ensurepip=yes \
    --with-system-expat \
    --with-system-ffi \
    --enable-loadable-sqlite-extensions \
    --with-dbmliborder=gdbm \
    --enable-unicode=ucs4

config:
  configure-commands:
  - ./configure %{conf-local}
  install-commands:
  - |
    %{make-install} DESTSHARED=/usr/lib/python2.7/lib-dynload
  - |
    rm -f "%{install-root}/usr/bin/2to3"
  - |
    rm -rf %{install-root}%{indep-libdir}/python2.7/lib2to3
  - |
    rm -rf %{install-root}/usr/bin/idle*
  - |
    rm -rf %{install-root}%{indep-libdir}/python2.7/idlelib
  - |
    rm -rf %{install-root}%{indep-libdir}/python2.7/lib-tk
  - |
    rm -rf %{install-root}%{indep-libdir}/python2.7/test
  - |
    rm -rf %{install-root}%{indep-libdir}/python2.7/*/test
  - |
    rm -rf %{install-root}%{indep-libdir}/python2.7/*/tests

  - |
    find "%{install-root}" -name "lib*.a" -exec rm {} ";"

  - |
    %{fix-pyc-timestamps}

sources:
- kind: tar
  url: https://github.com/python/cpython/archive/v2.7.17.tar.gz
  ref: d2b73d2d2607ec3f67bc10cef2d383008f95b835150d97cd00e55283d4afe360
  
depends:
- freedesktop-sdk.bst:bootstrap/openssl.bst
- freedesktop-sdk.bst:components/expat.bst
- freedesktop-sdk.bst:components/gdbm.bst
- freedesktop-sdk.bst:components/libffi.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/sqlite.bst
- freedesktop-sdk.bst:components/xz.bst
