kind: autotools
description: |

    The libnuma library offers a simple programming interface to the NUMA
    (Non Uniform Memory Access) policy supported by the Linux kernel. On
    a NUMA architecture some memory areas have different latency or
    bandwidth than others.

sources:
- kind: tar
  url: github:numactl/numactl/releases/download/v2.0.17/numactl-2.0.17.tar.gz
  ref: 8824ddd529a6f935dc2670bba6f52fb0f605f9f48e1176f69d6b7e48aad8cb27

depends:
- base.bst
