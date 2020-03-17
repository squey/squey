kind: cmake
description: |

    Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data.

sources:
- kind: tar
  url: https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.14/src/hdf5-1.8.14.tar.bz2
  ref: 53722cb9f5977af2908f92fc1a3f8c597800a5cb448cfe5c75060d6335bb9bab
  

variables:
  cmake-local: |
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=1 \
    -DHDF5_ENABLE_Z_LIB_SUPPORT=1 \
    -DHDF5_BUILD_CPP_LIB=1 \
    -DALLOW_UNSUPPORTED=1 \
    -DHDF5_ENABLE_THREADSAFE=1 \
    -G"Ninja"
    
config:
  configure-commands:
    - sed -i '18,20d' config/cmake/hdf5-config-version.cmake.in
    - "%{cmake}"

depends:
- base.bst
