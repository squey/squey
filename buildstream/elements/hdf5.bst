kind: cmake
description: |

    Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data.

sources:
- kind: tar
  url: https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.bz2
  ref: 97906268640a6e9ce0cde703d5a71c9ac3092eded729591279bf2e3ca9765f61

variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=1 \
    -DHDF5_ENABLE_Z_LIB_SUPPORT=1 \
    -DHDF5_BUILD_CPP_LIB=1 \
    -DALLOW_UNSUPPORTED=1 \
    -DHDF5_ENABLE_THREADSAFE=1 \
    -G"Ninja"

depends:
- base.bst
