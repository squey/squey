kind: cmake
description: |

    Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data.

sources:
- kind: git
  url: ssh://git@gitlab.esi-group.com/io/erf.git
  ref: 7f86c13e6c0d5e77128e9c42269f276c7bb645ec
  checkout-submodules: True
  submodules:
    cmake/ESICMakeModules:
      checkout: true
      url: git@gitlab.esi-group.com:Collaborative/ESICMakeModules.git
      
variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release \
    -DERF_MULTI_PRECISION=0 \
    -DERF_DOUBLE_PRECISION=0 \
    -DERF_STANDARD_ENCRYPTION=0 \
    -DERF_VPS_USER_ENCRYPTION=0 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -w" \
    -G"Ninja"
    
depends:
- base.bst
- hdf5.bst
build-depends:
- freedesktop-sdk.bst:components/git.bst
