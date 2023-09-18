kind: autotools
description: |

    The Portable Hardware Locality (hwloc) software package provides a portable abstraction (across OS, versions, architectures, ...) of the hierarchical topology of modern architectures, including NUMA memory nodes, sockets, shared caches, cores and simultaneous multithreading. It also gathers various system attributes such as cache and memory information as well as the locality of I/O devices such as network interfaces, InfiniBand HCAs or GPUs.

sources:
- kind: tar

  url: open-mpi:release/hwloc/v2.8/hwloc-2.8.0.tar.bz2
  ref: 348a72fcd48c32a823ee1da149ae992203e7ad033549e64aed6ea6eeb01f42c1

depends:
- base.bst
