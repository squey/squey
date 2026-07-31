kind: autotools
description: |

    Breakpad symbol tools, running on the build host.

    Crashpad reuses the Breakpad symbol format but ships no symbol dumper, so
    these are still needed. This fork is used rather than upstream because its
    dump_syms_dwarf reads DWARF out of PE/COFF executables, which is what
    mingw-w64 produces: upstream only reads PDB, through the Windows-only DIA
    SDK. dump_syms_mac reads Mach-O, so the three targets are covered from
    Linux.

    Note that squey has to be built with -gdwarf-4 for these to see the
    functions: they do not read the DWARF 5 .debug_rnglists section that recent
    GCC emits by default.

sources:
- kind: git
  url: github:DaemonEngine/breakpad.git
  ref: 3f088e97737b9915c4e2925e0cbfa82e6066cbc1

config:
  # Only the tools are wanted, not the client library, and building everything
  # would drag in the test suite.
  build-commands:
  - |
    make %{make-args} \
      src/tools/linux/dump_syms/dump_syms \
      src/tools/windows/dump_syms_dwarf/dump_syms_dwarf \
      src/tools/mac/dump_syms/dump_syms_mac \
      src/processor/minidump_stackwalk \
      src/processor/minidump_dump

  install-commands:
  - |
    install -D src/tools/linux/dump_syms/dump_syms -t "%{install-root}%{bindir}"
    install -D src/tools/windows/dump_syms_dwarf/dump_syms_dwarf -t "%{install-root}%{bindir}"
    install -D src/tools/mac/dump_syms/dump_syms_mac -t "%{install-root}%{bindir}"
    install -D src/processor/minidump_stackwalk -t "%{install-root}%{bindir}"
    install -D src/processor/minidump_dump -t "%{install-root}%{bindir}"

build-depends:
- base.bst
