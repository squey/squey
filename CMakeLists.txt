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

# Used for Dumbnet and Pcap
include(CheckFunctionExists)

# Used for ICU, TBB and HWLoc
include(FindPkgConfig)

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
add_custom_target(love COMMAND ${CMAKE_CTEST_COMMAND} -E BENCH)
add_dependencies(love update_tests_files)
add_dependencies(love testsuite)

###############################################################################
# Define project dependencies
###############################################################################

declare_internal_library(pvkernel)

# Helpers means various stuff we can use
# such as binaries or libraries, that depends
# on PVKernel
add_subdirectory(helpers)

declare_internal_library(inendi)
declare_internal_library(pvhive)
declare_internal_library(pvdisplays)
declare_internal_library(pvparallelview)
declare_internal_library(pvguiqt)

add_subdirectory(gui-qt)
add_subdirectory(scripts)

# Define generated files
###############################################################################
# Set inspector.sh and make_protect for current build
configure_file(inspector.sh.cmake inspector.sh @ONLY)
configure_file(make_protect.cmake make_protect @ONLY)
configure_file(grep_table.sh.cmake grep_table.sh @ONLY)

###############################################################################
# Define distribution stuff
###############################################################################
include(CMakePackage.txt)
include(CMakePackageDistribute.txt)
