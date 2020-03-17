kind: cmake
description: |

    mbed TLS (formerly known as PolarSSL) makes it trivially easy for developers to include
    cryptographic and SSL/TLS capabilities in their (embedded) products, facilitating this
    functionality with a minimal coding footprint.

sources:
- kind: tar

  url: https://tls.mbed.org/download/mbedtls-2.16.2-apache.tgz
  ref: a6834fcd7b7e64b83dfaaa6ee695198cb5019a929b2806cb0162e049f98206a4

variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_SHARED_MBEDTLS_LIBRARY=On \
    -DENABLE_TESTING=Off

depends:
- base.bst
