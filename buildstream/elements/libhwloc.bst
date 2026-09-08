kind: autotools
description: |

    The Portable Hardware Locality (hwloc) software package provides a portable abstraction (across OS, versions, architectures, ...) of the hierarchical topology of modern architectures, including NUMA memory nodes, sockets, shared caches, cores and simultaneous multithreading. It also gathers various system attributes such as cache and memory information as well as the locality of I/O devices such as network interfaces, InfiniBand HCAs or GPUs.

sources:
- kind: tar
  url: open-mpi:release/hwloc/v2.14/hwloc-2.14.0.tar.bz2
  ref: 966b9bb3e9f29f8d65ce8d106779e457f40e246a645e584b100772a42f9ae94b

(@): buildstream/elements/base/target.yml

(?):
- (target_triple == "x86_64-apple-darwin" or target_triple == "aarch64-apple-darwin"):
    config:
      install-commands:
        (>):
        - |
          %{toolchain-dir}/%{host}-install_name_tool -id "@rpath/libhwloc.15.%{dylib-extension}" "%{install-root}%{libdir}/libhwloc.%{dylib-extension}"

depends:
- sdk.bst
