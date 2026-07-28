kind: manual
description: |

    A parallel implementation of gzip for modern multi-processor, multi-core machines

# The build this replaces (kjkpub, 2.3.1-149) started its worker threads with a bare
# CreateThread, leaving the C runtime without its per-thread state. It deadlocked at
# startup before writing a single byte, and the export then hung forever waiting on a
# child that never returned -- it took the first run of all four series it was tried on.
# This build is the same pigz 2.3.1 but goes through _beginthreadex: 0 hangs in 20 runs.
sources:
- kind: remote
  url: sourceforge:project/pigz-for-windows/pigz-win32.zip
  ref: daf9a3000d3dc7666b3524a779bf64dd7cd21f33f3dc7755deabba54c04eee15

(@): buildstream/elements/base/target.yml

(?):
- target_triple == "x86_64-w64-mingw32":
    config:
      install-commands:
      - |
        7zzs x pigz-win32.zip
        install -D pigz.exe -t "%{install-root}%{bindir}"

build-depends:
- sdk.bst
- 7zip_host.bst
