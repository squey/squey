kind: cmake
description: |

    Apache Thrift is a lightweight, language-independent software stack for point-to-point RPC implementation.
    
sources:
- kind: tar
  url: github:apache/thrift/archive/refs/tags/v0.22.0.tar.gz
  ref: c4649c5879dd56c88f1e7a1c03e0fbfcc3b2a2872fb81616bffba5aa8a225a37

(@): buildstream/elements/base/target.bst

environment:
    CFLAGS: "-D_WINSOCK_DEPRECATED_NO_WARNINGS -fpermissive"
    CXXFLAGS: "-D_WINSOCK_DEPRECATED_NO_WARNINGS -fpermissive"

variables:
  source-dir: "."
  cmake-local: >-
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    -DBUILD_COMPILER=OFF
    -DWITH_PYTHON=OFF
    -DWITH_JAVA=OFF
    -DWITH_JAVASCRIPT=OFF
    -DWITH_NODEJS=OFF
    -DWITH_KOTLIN=OFF

config:
  (?):
  - (target_triple == "x86_64-w64-mingw32"):
      configure-commands:
        (<):
        - |
          find %{source-dir} \( -name TServerSocket.cpp -o -name Sync.h \) -exec sed 's/Windows.h/windows.h/' -i {} \;
          find %{source-dir} \( -name THttpServer.cpp -o -name TWebSocketServer.h \) -exec sed 's/Shlwapi.h/shlwapi.h/' -i {} \;
          find %{source-dir} -name SocketPair.cpp -exec sed 's/WS2tcpip.h/ws2tcpip.h/' -i {} \;
          find %{source-dir} -name TPipeServer.cpp -exec sed -e 's/AccCtrl.h/accctrl.h/' -e 's/Aclapi.h/aclapi.h/' -i {} \;

build-depends:
- sdk.bst

(?):
  - (target_triple != "x86_64-linux-gnu"):
      depends:
        - boost.bst
        - openssl.bst
        - zlib.bst
        - zstd.bst
        - lz4.bst
  - target_triple == "x86_64-linux-gnu":
      depends:
        - boost.bst
