cmake_minimum_required(VERSION 2.8)

project(picviz-inspector)

#set(CMAKE_VERBOSE_MAKEFILE ON)

include(CTest)

include(CheckIncludeFiles)
include(CheckFunctionExists)
include(CheckLibraryExists)

include(FindPkgConfig)

include(CMakeMacros.txt)
include(CMakeOptions.txt)
include(CMakeCompilers.txt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${picviz-inspector_SOURCE_DIR}/cmake)

set(USE_OPENMP true)

#
# Required libraries
#
include(CMakeRequiredLibraries.txt)

#
# Handles the version
#
include(CMakeVersionHandler.txt)

# OpenMP for everyone
add_definitions(${OpenMP_CXX_FLAGS})
message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")

#############################################
# BEGIN: ADD INCLUDE DIRS NEEDED EVERYWHERE #
#############################################
message(STATUS "QtCore include dirs: " ${QT_QTCORE_INCLUDE_DIR})
include_directories(AFTER ${QT_QTCORE_INCLUDE_DIR})
message(STATUS "Qt include dirs: " ${QT_INCLUDE_DIR})
include_directories(AFTER ${QT_INCLUDE_DIR})
#message(STATUS "Dumbnet include dirs: " ${DUMBNET_INCLUDE_DIRS})
#include_directories(AFTER ${DUMBNET_INCLUDE_DIRS})
message(STATUS "Boost include dirs: " ${BOOST_INCLUDEDIR})
include_directories(AFTER ${BOOST_INCLUDEDIR})
message(STATUS "TBB include dirs: " ${TBB_INCLUDE_DIRS})
include_directories(AFTER ${TBB_INCLUDE_DIRS})

# PVBase
include_directories(AFTER "${picviz-inspector_SOURCE_DIR}/include")

###########################################
# END: ADD INCLUDE DIRS NEEDED EVERYWHERE #
###########################################

# Test compilation target
ADD_CUSTOM_TARGET(alltests COMMAND make)

# Generic generated-ui files include directory
PV_QT4_UI_INC_DIR(QT_UI_INCLUDE_DIR)

add_subdirectory(scripts)
declare_internal_library(pvkernel)

# Helpers means various stuff we can use
# such as binaries or libraries, that depends
# on PVKernel
add_subdirectory(helpers)

declare_internal_library(picviz)
declare_internal_library(pvhive)
declare_internal_library(pvdisplays)
declare_internal_library(pvparallelview)
declare_internal_library(pvguiqt)

add_subdirectory(gui-qt)


message(STATUS "CUDA found: " ${CUDA_FOUND})
if(CUDA_FOUND)
  message(STATUS "CUDA version: " ${CUDA_VERSION_STRING})
  message(STATUS "CUDA toolkit root dir: " ${CUDA_TOOLKIT_ROOT_DIR})
  message(STATUS "CUDA SDK root dir: " ${CUDA_SDK_ROOT_DIR})
  message(STATUS "CUDA 64bit device: " ${CUDA_64_BIT_DEVICE_CODE})
endif(CUDA_FOUND)

# Set inspector.sh and make_protect for current build
configure_file(inspector.sh.cmake inspector.sh @ONLY)
configure_file(make_protect.cmake make_protect @ONLY)

# generate tests scripts
configure_file(squish_run.sh.cmake squish_run.sh @ONLY)
configure_file(squish_ide.sh.cmake squish_ide.sh @ONLY)

include(CMakePackage.txt)
include(CMakePackageDistribute.txt)
