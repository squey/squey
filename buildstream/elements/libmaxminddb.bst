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

  url: https://github.com/maxmind/libmaxminddb/archive/1.3.2.tar.gz
  ref: c0785521c7e5515f1169db90ed6e51bc2a5a000377d0fbad87e4d5a791a6e364

config:

  # Override autotools element default configure-commands
  configure-commands:
  - |
    ./bootstrap
    %{configure} --disable-tests

depends:
- base.bst
- c-ares.bst
