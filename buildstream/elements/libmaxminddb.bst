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

  url: github:maxmind/libmaxminddb/archive/1.7.1.tar.gz
  ref: 4ed9c409667f0adeb3eca653780785ad04713b8a87f2fcfbb31967b213826322

config:

  # Override autotools element default configure-commands
  configure-commands:
  - |
    ./bootstrap
    %{configure} --disable-tests

depends:
- base.bst
- freedesktop-sdk.bst:components/c-ares.bst
