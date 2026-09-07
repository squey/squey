kind: cmake
description: zlib

sources:
- kind: git_repo
  url: github:madler/zlib.git
  track: v*
  ref: v1.3.2-0-gda607da739fa6047df13e66a2af6b8bec7c2a498

(@): buildstream/elements/base/target.yml

variables:
  # infcover, one of zlib's test targets, links with -coverage, which mingw's
  # libgcov cannot satisfy (__gcov_exit). The tests are of no use here anyway.
  cmake-local: >-
    -DZLIB_BUILD_TESTING=OFF

(?):
- (target_triple == "x86_64-linux-gnu"):
    config:
      install-commands:
        (>):
        - |
          find "%{install-root}" -name "lib*.a" -delete
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libz.1.%{dylib-extension}" "%{install-root}%{libdir}/libz.%{dylib-extension}"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/libz.%{dylib-extension}'
  cpe:
    vendor: zlib

build-depends:
- sdk.bst
