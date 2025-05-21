kind: cmake

(@): buildstream/elements/mysql.yml

variables:
  cmake-local-extra: >-
    -DCMAKE_CXX_COMPILER="g++"
    -DCMAKE_C_COMPILER="gcc"

config:
  install-commands:
  - |
    %{make-install}
    cp %{build-dir}/runtime_output_directory/* %{install-root}%{bindir}

public:
  bst:
    split-rules:
      devel:
        (>):
        - '%{bindir}/cno_huffman_generator'
        - '%{bindir}/comp_client_err'
        - '%{bindir}/uca9dump'

depends:
- base.bst
- freedesktop-sdk.bst:components/openssl.bst
