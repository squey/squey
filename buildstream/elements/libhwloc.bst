kind: autotools
description: |

    The Portable Hardware Locality (hwloc) software package provides a portable abstraction (across OS, versions, architectures, ...) of the hierarchical topology of modern architectures, including NUMA memory nodes, sockets, shared caches, cores and simultaneous multithreading. It also gathers various system attributes such as cache and memory information as well as the locality of I/O devices such as network interfaces, InfiniBand HCAs or GPUs.

sources:
- kind: tar
  url: open-mpi:release/hwloc/v2.11/hwloc-2.11.1.tar.bz2
  ref: 04cdfbffad225ce15f66184f0f4141327dabf288d10a8b84d13f517acb7870c6

(@): buildstream/elements/base/target.bst

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libhwloc.15.%{dylib-extension}" "%{install-root}%{libdir}/libhwloc.%{dylib-extension}"

depends:
- sdk.bst
