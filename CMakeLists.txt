#
# @file
#
# @copyright (C) Picviz Labs 2011-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

cmake_minimum_required(VERSION 3.3) # 3.4 break the build

project(inendi-inspector)

cmake_policy(SET CMP0002 OLD)

# this policy remains to "OLD" until the release process uses cmake 3.x
if(POLICY  CMP0026)
	cmake_policy(SET CMP0026 OLD)
endif()

###############################################################################
# Configure linker
###############################################################################
# Use ld.gold if available
OPTION(USE_LD_GOLD "Use GNU gold linker" ON)
if (USE_LD_GOLD)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
    if ("${LD_VERSION}" MATCHES "GNU gold")
        message(STATUS "using GNU gold linker as it is available.")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
    else ()
        message(WARNING "GNU gold linker isn't available, using the default system linker.")
        set(USE_LD_GOLD OFF)
    endif ()
endif () 

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

####################n###########################################################
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
# Define tests targets
###############################################################################
# Declare test suite target
# It has been populated thanks to the declare_pv_test macro

add_custom_target(inspector_testsuite)
if (CMAKE_CONFIGURATION_TYPES)
    add_custom_target(inspector_run_testsuite COMMAND ${CMAKE_CTEST_COMMAND} -R INSPECTOR_TEST_
        --force-new-ctest-process --output-on-failure
        --build-config "$<CONFIGURATION>")
else()
    add_custom_target(inspector_run_testsuite COMMAND ${CMAKE_CTEST_COMMAND} -R INSPECTOR_TEST_
        --force-new-ctest-process --output-on-failure)
endif()

add_dependencies(inspector_run_testsuite inspector_testsuite)
add_custom_target(inspector_bench COMMAND ${CMAKE_CTEST_COMMAND} -R INSPECTOR_BENCH_)
add_dependencies(inspector_bench inspector_testsuite)

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
# inendi-utils
###############################################################################
SET(PVLOGGER_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/inendi-utils/libpvlogger/src/libpvlogger.so)
SET(PVHWLOC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/inendi-utils/libpvhwloc/src/libpvhwloc.so)
SET(PVLICENSE_LIBRARIES
    ${CMAKE_SOURCE_DIR}/inendi-utils/core/license/lib/flexnet/libesiflex.a
    ${CMAKE_SOURCE_DIR}/inendi-utils/core/license/lib/sentinel/liblssrv64.a
    dl
)
include_directories(${CMAKE_SOURCE_DIR}/inendi-utils/core/license/include)
include_directories(${CMAKE_SOURCE_DIR}/inendi-utils/libpvlogger/include)
include_directories(${CMAKE_SOURCE_DIR}/inendi-utils/libpvhwloc/include)
add_subdirectory(inendi-utils)

###############################################################################
# libpvcop
###############################################################################
add_subdirectory(libpvcop)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libpvcop/include)
SET(PVCOP_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libpvcop/src/libpvcop.so)

declare_internal_library(pvkernel)
declare_internal_library(inendi)
declare_internal_library(pvdisplays)
declare_internal_library(pvparallelview)
declare_internal_library(pvguiqt)

add_subdirectory(gui-qt)

# Define generated files
###############################################################################
# Set inspector.sh for current build
configure_file(inspector.sh.cmake inspector.sh @ONLY)
