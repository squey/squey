#
# @file
#

cmake_minimum_required(VERSION 3.26.4)

project(squey)

###############################################################################
# Define testing environment
###############################################################################
include(CTest)
enable_testing()

###############################################################################
# Generic configuration for squey
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
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${squey_SOURCE_DIR}/cmake)

# FIXME : This is a Hack waiting for QWebEngine integration
set(QT_WEBKIT 0)
if(${QT_WEBKIT})
	add_definitions(-DQT_WEBKIT)
	message(STATUS "Using QtWebKit")
else(${QT_WEBKIT})
	message(STATUS "Using QtWebEngine")
endif(${QT_WEBKIT})

# Required libraries
# Define library that will be used everywhere in Squey
include(CMakeRequiredLibraries.txt)

# Handles the version
include(CMakeVersionHandler.txt)

# Define variable and repo required for tests
include(CMakePVTests.txt)

####################n###########################################################
# Define distribution stuff
###############################################################################
include(CMakePackage.txt)
include(CMakePackageDistribute.txt)

###############################################################################
# Add generic squey includes
###############################################################################

# PVBase
include_directories(AFTER "${squey_SOURCE_DIR}/include")

###############################################################################
# Define tests targets
###############################################################################
# Declare test suite target
# It has been populated thanks to the declare_pv_test macro

add_custom_target(squey_testsuite)
if (CMAKE_CONFIGURATION_TYPES)
    add_custom_target(squey_run_testsuite COMMAND ${CMAKE_CTEST_COMMAND} -R SQUEY_TEST_
        --force-new-ctest-process --output-on-failure
        --build-config "$<CONFIGURATION>")
else()
    add_custom_target(squey_run_testsuite COMMAND ${CMAKE_CTEST_COMMAND} -R SQUEY_TEST_
        --force-new-ctest-process --output-on-failure)
endif()

add_dependencies(squey_testsuite squey)
add_dependencies(squey_run_testsuite squey_testsuite)
add_custom_target(squey_bench COMMAND ${CMAKE_CTEST_COMMAND} -R SQUEY_BENCH_)
add_dependencies(squey_bench squey_testsuite)

###############################################################################
# Define project dependencies
###############################################################################
include(ExternalProject)

###############################################################################
# furl
###############################################################################
set(FURL_STATIC_LIB ${CMAKE_CURRENT_BINARY_DIR}/furl-prefix/src/furl-build/src/lib/libfurll-utf8.a)
ExternalProject_Add(furl
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/furl
    STEP_TARGETS build
  	EXCLUDE_FROM_ALL TRUE
  	CMAKE_ARGS -DARCH_TYPE=${ARCH_TYPE}
  	BUILD_BYPRODUCTS ${FURL_STATIC_LIB}
  	INSTALL_COMMAND cmake -E echo "Skipping install step."
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/furl/src/lib/include)
add_definitions(-DFURL_UTF8_CHAR)

###############################################################################
# squey-utils
###############################################################################
SET(PVLOGGER_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/squey-utils/libpvlogger/src/libpvlogger.so)
SET(PVHWLOC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/squey-utils/libpvhwloc/src/libpvhwloc.so)
include_directories(${CMAKE_SOURCE_DIR}/squey-utils/libpvlogger/include)
include_directories(${CMAKE_SOURCE_DIR}/squey-utils/libpvhwloc/include)
add_subdirectory(squey-utils)

###############################################################################
# libpvcop
###############################################################################
add_subdirectory(libpvcop)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libpvcop/include)
SET(PVCOP_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libpvcop/src/libpvcop.so)

declare_internal_library(pvkernel)
declare_internal_library(squey)
declare_internal_library(pvdisplays)
declare_internal_library(pvparallelview)
declare_internal_library(pvguiqt)

add_subdirectory(gui-qt)

# Define generated files
###############################################################################
# Set squey.sh for current build
configure_file(squey.sh.cmake squey.sh @ONLY)
