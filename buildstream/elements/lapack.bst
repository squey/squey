kind: cmake
description: |

    LAPACK is a library of Fortran subroutines for solving the most commonly occurring problems in numerical linear algebra.

sources:
- kind: tar
  url: https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz
  ref: 106087f1bb5f46afdfba7f569d0cbe23dacb9a07cd24733765a0e89dbe1ad573
  
environment:
  INTERFACE64: 1
  SYMBOLSUFFIX: "64_"

depends:
- base.bst
