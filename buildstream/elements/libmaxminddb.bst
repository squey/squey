kind: autotools
description: |

    The libmaxminddb library provides a C library for reading MaxMind DB files, including the GeoIP2 databases from MaxMind.
    This is a custom binary format designed to facilitate fast lookups of IP addresses while allowing for great flexibility
    in the type of data associated with an address.

    The MaxMind DB format is an open format. The spec is available at http://maxmind.github.io/MaxMind-DB/.
    This spec is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License.

    See http://dev.maxmind.com/ for more details about MaxMind`s GeoIP2 products.

sources:
- kind: tar
  url: github:maxmind/libmaxminddb/archive/refs/tags/1.11.0.tar.gz
  ref: dfda7a1367b64b78acd99ac309d6e4bcfea3408678808d3f48bf41728b321909

(@): buildstream/elements/base/target.bst

variables:
  conf-local: |
    --disable-tests

depends:
- sdk.bst
- c-ares.bst
- freedesktop-sdk.bst:components/pkg-config.bst