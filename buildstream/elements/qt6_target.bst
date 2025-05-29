kind: cmake

(@):
  - buildstream/elements/base/target.bst
  - buildstream/elements/qt6.yml

(?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      variables:
        cmake-local-extra: >-
          -DQT_FORCE_WARN_APPLE_SDK_AND_XCODE_CHECK=ON
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      environment:
        CXXFLAGS: "-fuse-ld=%{toolchain-dir}/%{host}-ld"
  - (target_triple == "x86_64-w64-mingw32"):
      variables:
        cmake-local-extra: >-
          -DQT_FEATURE_stack_clash_protection=OFF

build-depends:
  (>):
  - sdk.bst
  - qt6_host.bst

depends:
  (>):
  - mariadb-connector-c_target.bst
  - libpq_target.bst
  - brotli.bst
  - icu.bst
  - pcre2.bst
  - openssl.bst
  - zstd.bst