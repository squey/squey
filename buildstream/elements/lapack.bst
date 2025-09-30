kind: cmake
description: |

    LAPACK is a library of Fortran subroutines for solving the most commonly occurring problems in numerical linear algebra.

sources:
- kind: tar
  url: github:Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz
  ref: 2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422

environment:
  INTERFACE64: 1
  SYMBOLSUFFIX: "64_"

depends:
- base.bst
