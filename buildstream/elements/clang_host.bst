kind: cmake

(@):  buildstream/elements/clang.yml

variables:
  cmake-local-extra: >-
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DLLVM_DEFAULT_TARGET_TRIPLE=%{build-triplet}
  (?):
  - target_triple == "x86_64-linux-gnu":
      # Mesa's rusticl, which the GL runtime brings to drive AMD and Intel
      # GPUs, initialises every backend of the LLVM it was built against, BPF
      # among them. That LLVM and this one share a soname, so whichever loads
      # first serves both, and it is often this one: the RPATH of PortableCL
      # names /app/lib, and run_cmd.sh puts it first on the search path. Without
      # BPF here, rusticl failed to load.
      llvm-targets: "X86;AArch64;AMDGPU;BPF;NVPTX"

depends:
  (>):
  - base.bst
