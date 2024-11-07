kind: meson

sources:
- kind: git_repo
  url: github:lz4/lz4.git
  track: v*
  ref: v1.10.0-0-gebb370ca83af193212df4dcbadcc5d87bc0de2f0

(@): buildstream/elements/base/target.bst

variables:
  command-subdir: build/meson
  meson-local: -Dprograms=true -Dtests=false -Dossfuzz=false -Dexamples=false --cross-file %{host}.txt

config:
  configure-commands:
  - |
    cat <<EOF > "%{host}.txt"
    [binaries]
    name = '%{target-platform}'
    c = '%{host}-clang'
    c_ld = '%{toolchain-dir}/%{host}-ld'
    ar = '%{host}-ar'
    ranlib = '%{host}-ranlib'
    strip = '%{host}-strip'

    [host_machine]
    system = '%{target-platform}'
    cpu_family = '%{target-arch}'
    cpu = '%{target-arch}'
    endian = 'little'
    EOF
  - "%{meson}"

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{libdir}/liblz4.%{dylib-extension}'

  cpe:
    vendor: lz4_project

build-depends:
- sdk.bst
- freedesktop-sdk.bst:components/meson.bst