kind: autotools
description: |

    The Portable Hardware Locality (hwloc) software package provides a portable abstraction (across OS, versions, architectures, ...) of the hierarchical topology of modern architectures, including NUMA memory nodes, sockets, shared caches, cores and simultaneous multithreading. It also gathers various system attributes such as cache and memory information as well as the locality of I/O devices such as network interfaces, InfiniBand HCAs or GPUs.

sources:
- kind: tar

  url: https://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-1.11.8.tar.bz2
  ref: ecef10265084d9123dc0b03d82ff5ee06fbf5ba755a912e7511ae590c9ee594d

depends:
- base.bst
