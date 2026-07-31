kind: manual
description: |

    JWasm is a MASM-compatible assembler. It runs on the build host and is only
    needed to assemble the two MASM sources of the Windows Crashpad handler,
    which mingw-w64 cannot assemble itself.

sources:
- kind: git
  url: github:Baron-von-Riedesel/JWasm.git
  ref: 7f6f32e78b79565d40bcce496756aadd1ff66900

config:
  build-commands:
  - make -f GccUnix.mak

  install-commands:
  - install -D build/GccUnixR/jwasm -t "%{install-root}%{bindir}"

build-depends:
- base.bst
