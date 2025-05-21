kind: cmake
description: |

    Syntax highlighting engine for Kate syntax definitions

sources:
- kind: git
  url: github:KDE/syntax-highlighting
  ref: 4dae8bf4ad2fd018f9c56450d9f09a32ec0f99af

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
