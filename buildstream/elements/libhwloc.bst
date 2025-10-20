kind: autotools
description: |

    The Portable Hardware Locality (hwloc) software package provides a portable abstraction (across OS, versions, architectures, ...) of the hierarchical topology of modern architectures, including NUMA memory nodes, sockets, shared caches, cores and simultaneous multithreading. It also gathers various system attributes such as cache and memory information as well as the locality of I/O devices such as network interfaces, InfiniBand HCAs or GPUs.

sources:
- kind: tar
  url: open-mpi:release/hwloc/v2.12/hwloc-2.12.2.tar.bz2
  ref: 563e61d70febb514138af0fac36b97621e01a4aacbca07b86e7bd95b85055ba0

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
