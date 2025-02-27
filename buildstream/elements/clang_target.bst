kind: cmake

(@):
  - buildstream/elements/base/target.bst
  - buildstream/elements/clang.yml

variables:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      cmake-local-extra: >-
        -DLLVM_ENABLE_PROJECTS='clang;compiler-rt;lld'
        -DLLVM_DEFAULT_TARGET_TRIPLE=%{target_triple}
        -DCMAKE_CXX_FLAGS="-fuse-ld=%{toolchain-dir}/%{host}-ld"
  - target_triple == "x86_64-w64-mingw32":
      cmake-local-extra: >-
        -DBUILD_SHARED_LIBS:BOOL=ON
        -DLLVM_LINK_LLVM_DYLIB:BOOL=OFF
        -DLLVM_BINUTILS_INCDIR=""
        -DCLANG_LINK_CLANG_DYLIB=OFF
        -DLLVM_TARGETS_TO_BUILD="X86"
        -DCMAKE_CXX_FLAGS="-fvisibility-inlines-hidden -Wl,--allow-multiple-definition -s -Wa,-mbig-obj"

config:
  (?):
  - (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -rpath %{libdir} @executable_path/../Frameworks %{install-root}%{bindir}/lld

build-depends:
  (>):
  - sdk.bst
  - binutils.bst