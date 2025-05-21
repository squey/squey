kind: cmake
description: |

    Extra modules and scripts for CMake.

sources:
- kind: git
  url: kde:kitemmodels.git
  ref: 3a6b78f7f292330ecb90966974eebf8e596316b4

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
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libKF6ItemModels.6.%{dylib-extension}" "%{install-root}%{libdir}/libKF6ItemModels.6.%{dylib-extension}"
- target_triple != "x86_64-linux-gnu":
    depends:
      - qt6_target.bst
