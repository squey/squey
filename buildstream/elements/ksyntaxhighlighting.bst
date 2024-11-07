kind: cmake
description: |

    Syntax highlighting engine for Kate syntax definitions

sources:
- kind: tar
  url: github:KDE/syntax-highlighting/archive/refs/tags/v6.5.0.tar.gz
  ref: 2c754abe5c226a627f6064096b7d187552e4055a593d689f3fc68a3b5903e2fd

(@): buildstream/elements/base/target.bst

variables:
  cmake-local: >-
    -DCMAKE_PREFIX_PATH="%{prefix}"
    -DQT_HOST_PATH="%{host-prefix}"
    -DQT_MAJOR_VERSION=6
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fpermissive"

build-depends:
- sdk.bst
- qt6_host.bst
- cmake-extra-modules.bst

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libKF6SyntaxHighlighting.6.%{dylib-extension}" "%{install-root}%{libdir}/libKF6SyntaxHighlighting.6.%{dylib-extension}"
- target_triple != "x86_64-linux-gnu":
    depends:
      - qt6_target.bst
