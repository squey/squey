kind: cmake
description: |

    Qt for Python is the project that provides the official set of Python bindings (PySide2) that will supercharge your Python applications.

sources:
- kind: tar
  url: https://download.qt.io/official_releases/QtForPython/pyside2/PySide2-5.13.1-src/pyside-setup-everywhere-src-5.13.1.tar.xz
  ref: c57c71ddece90b29d0934cf5fac1e1634719484637b56dcbe49eb277c1c01536

- kind: patch
  path: buildstream/patches/pyside2/pyside2.patch
  strip-level: 1
  
variables:
  cmake-local: |
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DMODULES=0

config:
  install-commands:
    - |
      env DESTDIR="%{install-root}" cmake --build %{build-dir} --target install
      mv %{install-root}%{prefix}/share/PySide2/* "%{install-root}%{prefix}/lib/python3.7/site-packages/PySide2/"
    
build-depends:
- base.bst
- qt5.bst
- freedesktop-sdk.bst:components/git.bst

depends:
- freedesktop-sdk.bst:components/libxslt.bst
- freedesktop-sdk.bst:components/python3.bst
- freedesktop-sdk.bst:components/llvm.bst


 
