cmake_minimum_required(VERSION 2.8)

project(Picviz_Inspector)

if (WIN32)
	#set(CMAKE_BUILD_TYPE "Debug")
	set(CMAKE_BUILD_TYPE "Release")
	#set(CMAKE_BUILD_TYPE "Debug")
else(WIN32)
	set(CMAKE_BUILD_TYPE "Debug")
endif(WIN32)

set(PICVIZ_SINGLE_TREE_BUILD true)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(CTest)

include(CheckIncludeFiles)
include(CheckFunctionExists)
include(CheckLibraryExists)

include(FindPkgConfig)

#
# Application specific definitions
#
add_definitions(-DCUSTOMER_RELEASE)

if(WIN32)
	add_definitions(-DWIN32)
	add_definitions(-D_WIN32)
	add_definitions(-D_WINSOCKAPI_)
else(WIN32)
	add_definitions(-DUNIX)
	add_definitions(-DLINUX=2)
	add_definitions(-Wall -Wextra -pipe -ggdb3)
endif(WIN32)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${Picviz_Inspector_SOURCE_DIR}/cmake)


set(USE_OPENMP true)
#########################################
# BEGIN: SET VARIOUS REQUIRED LIBRARIES #
#########################################
include(CMakeRequiredLibraries.txt)
#######################################
# END: SET VARIOUS REQUIRED LIBRARIES #
#######################################

#############################################
# BEGIN: ADD INCLUDE DIRS NEEDED EVERYWHERE #
#############################################
message(STATUS "QtCore include dirs: " ${QT_QTCORE_INCLUDE_DIR})
include_directories(AFTER ${QT_QTCORE_INCLUDE_DIR})
message(STATUS "Qt include dirs: " ${QT_INCLUDE_DIR})
include_directories(AFTER ${QT_INCLUDE_DIR})
message(STATUS "Dumbnet include dirs: " ${DUMBNET_INCLUDE_DIRS})
include_directories(AFTER ${DUMBNET_INCLUDE_DIRS})
message(STATUS "Boost include dirs: " ${BOOST_INCLUDEDIR})
include_directories(AFTER ${BOOST_INCLUDEDIR})
message(STATUS "TBB include dirs: " ${TBB_INCLUDE_DIRS})
include_directories(AFTER ${TBB_INCLUDE_DIRS})
###########################################
# END: ADD INCLUDE DIRS NEEDED EVERYWHERE #
###########################################


add_subdirectory(scripts)
add_subdirectory(libpvcore)
add_subdirectory(libpvfilter)
add_subdirectory(libpvrush)
add_subdirectory(libpicviz)
add_subdirectory(libpvgl)
add_subdirectory(gui-qt)

message(STATUS "CUDA found: " ${CUDA_FOUND})
if(CUDA_FOUND)
  message(STATUS "CUDA version: " ${CUDA_VERSION_STRING})
  message(STATUS "CUDA toolkit root dir: " ${CUDA_TOOLKIT_ROOT_DIR})
  message(STATUS "CUDA SDK root dir: " ${CUDA_SDK_ROOT_DIR})
  message(STATUS "CUDA 64bit device: " ${CUDA_64_BIT_DEVICE_CODE})
endif(CUDA_FOUND)

#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/src/include/picviz/config.h)


include(CMakePackage.txt)
include(CMakePackageDistribute.txt)

	
