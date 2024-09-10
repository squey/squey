kind: cmake
description: |

    LAPACK is a library of Fortran subroutines for solving the most commonly occurring problems in numerical linear algebra.

sources:
- kind: tar
  url: github:Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
  ref: eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b
  
environment:
  INTERFACE64: 1
  SYMBOLSUFFIX: "64_"

depends:
- base.bst
