kind: autotools
description: |

    The libnuma library offers a simple programming interface to the NUMA
    (Non Uniform Memory Access) policy supported by the Linux kernel. On
    a NUMA architecture some memory areas have different latency or
    bandwidth than others.

sources:
- kind: tar
  url: github:numactl/numactl/releases/download/v2.0.19/numactl-2.0.19.tar.gz
  ref: f2672a0381cb59196e9c246bf8bcc43d5568bc457700a697f1a1df762b9af884

depends:
- base.bst
