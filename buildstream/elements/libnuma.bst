kind: autotools
description: |

    The libnuma library offers a simple programming interface to the NUMA
    (Non Uniform Memory Access) policy supported by the Linux kernel. On
    a NUMA architecture some memory areas have different latency or
    bandwidth than others.

sources:
- kind: tar

  url: github:numactl/numactl/releases/download/v2.0.13/numactl-2.0.13.tar.gz
  ref: 991e254b867eb5951a44d2ae0bf1996a8ef0209e026911ef6c3ef4caf6f58c9a

depends:
- base.bst
