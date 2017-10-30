#
# @file
#
# @copyright (C) Picviz Labs 2011-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

cmake_minimum_required(VERSION 2.8)

project(inendi-inspector)

#set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_policy(SET CMP0002 OLD)

# this policy remains to "OLD" until the release process uses cmake 3.x
if(POLICY  CMP0026)
	cmake_policy(SET CMP0026 OLD)
endif()

###############################################################################
# Define testing environment
###############################################################################
include(CTest)
enable_testing()

###############################################################################
# Generic configuration for inspector
###############################################################################

# Used for ICU, TBB and HWLoc
find_package(PkgConfig)

# User defined macro
include(CMakeMacros.txt)

# Define option related to enabled/disabled features
include(CMakeOptions.txt)

# Define compilation option such CPP/CXXFLAGS, libs ...
include(CMakeCompilers.txt)

# Define folder for find_package macro
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${inendi-inspector_SOURCE_DIR}/cmake)

# FIXME : This is a Hack waiting for QWebEngine integration
set(QT_WEBKIT 1)
if(${QT_WEBKIT})
	add_definitions(-DQT_WEBKIT)
	message(STATUS "Using QtWebKit")
else(${QT_WEBKIT})
	message(STATUS "Using QtWebEngine")
endif(${QT_WEBKIT})

# Required libraries
# Define library that will be used everywhere in Inspector
include(CMakeRequiredLibraries.txt)

# Handles the version
include(CMakeVersionHandler.txt)

# Define variable and repo required for tests
include(CMakePVTests.txt)

###############################################################################
# Define distribution stuff
###############################################################################
include(CMakePackage.txt)
include(CMakePackageDistribute.txt)

###############################################################################
# Add generic inspector includes
###############################################################################

# PVBase
include_directories(AFTER "${inendi-inspector_SOURCE_DIR}/include")

###############################################################################
# Define svn update tests files target
###############################################################################
# TODO : add builder account on SVN machine auth by ssh
add_custom_target(update_tests_files COMMAND "${CMAKE_SOURCE_DIR}/tests/update-tests-files.sh" "${TESTS_FILES_DIR}" "${CMAKE_SOURCE_DIR}/tests/files-svn-rev")

###############################################################################
# Define tests targets
###############################################################################
# Declare test suite target
# It has been populated thanks to the declare_pv_test macro
# `make testsuite` will compile tests
# `make love` will compile, download ref files and then run tests
add_custom_target(testsuite)
add_custom_target(love COMMAND ${CMAKE_CTEST_COMMAND} -E \"BENCH|LONG\")
add_dependencies(love update_tests_files)
add_dependencies(love testsuite)
add_custom_target(bench COMMAND ${CMAKE_CTEST_COMMAND} -R BENCH)
add_dependencies(bench update_tests_files)
add_dependencies(bench testsuite)

###############################################################################
# Define project dependencies
###############################################################################

cmake_minimum_required(VERSION 2.8)
include(ExternalProject)


###############################################################################
# furl
###############################################################################

ExternalProject_Add(furl
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/furl
    BUILD_IN_SOURCE 1
    STEP_TARGETS build
  	EXCLUDE_FROM_ALL TRUE
  	INSTALL_COMMAND cmake -E echo "Skipping install step."
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/furl/src/lib/include)
set(FURL_STATIC_LIB ${CMAKE_CURRENT_SOURCE_DIR}/furl/src/lib/libfurll-utf8.a)
add_definitions(-DFURL_UTF8_CHAR)

###############################################################################
# inendi-utils
###############################################################################
pkg_check_modules(HWLOC REQUIRED hwloc)
ExternalProject_Add(inendi-utils
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils
    BUILD_IN_SOURCE 1
    STEP_TARGETS build
  	EXCLUDE_FROM_ALL TRUE
  	INSTALL_COMMAND cmake -E echo "Skipping install step."
)
SET(PVLOGGER_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/libpvlogger/src/libpvlogger.so)
SET(PVHWLOC_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/libpvhwloc/src/libpvhwloc.so)
SET(PVLICENSE_LIBRARIES
    ${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/core/license/lib/flexnet/libesiflex.a
    ${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/core/license/lib/sentinel/liblssrv64.a
    dl
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/libpvlogger/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/libpvhwloc/include)
SET(PVLICENSE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inendi-utils/core/license/include)

###############################################################################
# libpvcop
###############################################################################

ExternalProject_Add(libpvcop
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libpvcop
    BUILD_IN_SOURCE 1
    STEP_TARGETS build
  	EXCLUDE_FROM_ALL TRUE
  	INSTALL_COMMAND cmake -E echo "Skipping install step."
)
SET(PVCOP_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/libpvcop/src/libpvcop.so)
MESSAGE(STATUS PVCOP_LIBRARIES=${PVCOP_LIBRARIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libpvcop/include)

declare_internal_library(pvkernel)
declare_internal_library(inendi)
declare_internal_library(pvdisplays)
declare_internal_library(pvparallelview)
declare_internal_library(pvguiqt)

add_dependencies(libpvcop inendi-utils)
add_dependencies(pvkernel furl libpvcop)

add_subdirectory(gui-qt)

# Define generated files
###############################################################################
# Set inspector.sh for current build
configure_file(inspector.sh.cmake inspector.sh @ONLY)
