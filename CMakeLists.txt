cmake_minimum_required(VERSION 2.8)

project(Picviz_Inspector)

if(EXISTS "use_pvgl2")
  set(USE_PVGL2 true)
  message(STATUS "Use PVGL version: 2")
else(EXISTS "use_pvgl2")
  message(STATUS "Use PVGL version: 1")
endif(EXISTS "use_pvgl2")

if(USE_PVGL2)
  add_definitions(-DUSE_PVGL2)
endif(USE_PVGL2)

if (WIN32)
	#set(CMAKE_BUILD_TYPE "Debug")
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
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

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
	if(WIN32)
		# /Ox: "full-optimisation" mode
		# /Ob2: inline everything that is possible
		# /Oi: use intrinsics functions
		# /Ot: favorize fast over small code
		# /GL: intra-procedural optimisations by MSVC !
		set(COMPILER_OPTIMIZE_FLAGS "/Ox /Ob2 /Oi /Ot /GL")
	else(WIN32)
		set(COMPILER_OPTIMIZE_FLAGS "-O3 -march=native")
	endif(WIN32)
	add_definitions(-DNDEBUG) # so that 'assert' will generate no code
endif()

#
# Application specific definitions
#
add_definitions(-DCUSTOMER_RELEASE)

if(WIN32)
	add_definitions(-DWIN32)
	add_definitions(-D_WIN32)
	add_definitions(-D_WINSOCKAPI_)
	set(MSVC_LINKER_FLAGS "/LARGEADDRESSAWARE")
	if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		# Intraprocedural optimisations
		set(MSVC_LINKER_FLAGS "${MSVC_LINKER_FLAGS} /ltcg")
	endif()
	set(CMAKE_EXE_LINKER_FLAGS ${MSVC_LINKER_FLAGS})
	set(CMAKE_SHARED_LINKER_FLAGS ${MSVC_LINKER_FLAGS})
	set(CMAKE_MODULE_LINKER_FLAGS ${MSVC_LINKER_FLAGS})

else(WIN32)
	add_definitions(-DUNIX)
	add_definitions(-DLINUX=2)
	add_definitions(-Wall -Wextra -pipe -ggdb3 -msse4.1)
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
#message(STATUS "Dumbnet include dirs: " ${DUMBNET_INCLUDE_DIRS})
#include_directories(AFTER ${DUMBNET_INCLUDE_DIRS})
message(STATUS "Boost include dirs: " ${BOOST_INCLUDEDIR})
include_directories(AFTER ${BOOST_INCLUDEDIR})
message(STATUS "TBB include dirs: " ${TBB_INCLUDE_DIRS})
include_directories(AFTER ${TBB_INCLUDE_DIRS})

# PVBase
include_directories(AFTER "${Picviz_Inspector_SOURCE_DIR}/include")

###########################################
# END: ADD INCLUDE DIRS NEEDED EVERYWHERE #
###########################################


add_subdirectory(scripts)
add_subdirectory(libpvkernel)
add_subdirectory(libpicviz)

if(USE_PVGL2)
  add_subdirectory(libpvgl2)
else(USE_PVGL2)
  add_subdirectory(libpvgl)
endif(USE_PVGL2)

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

	
