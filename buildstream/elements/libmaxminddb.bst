kind: autotools
description: |

    The libmaxminddb library provides a C library for reading MaxMind DB files, including the GeoIP2 databases from MaxMind.
    This is a custom binary format designed to facilitate fast lookups of IP addresses while allowing for great flexibility
    in the type of data associated with an address.

    The MaxMind DB format is an open format. The spec is available at http://maxmind.github.io/MaxMind-DB/.
    This spec is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License.

    See http://dev.maxmind.com/ for more details about MaxMind's GeoIP2 products.

sources:
- kind: tar

  url: github:maxmind/libmaxminddb/releases/download/1.8.0/libmaxminddb-1.8.0.tar.gz
  ref: 1107799f77be6aa3b9796ad0eed8ffcc334bf45f8bd18e6a984d8adf3e719c6d

variables:
  conf-local: |
    --disable-tests

depends:
- base.bst
- c-ares.bst
