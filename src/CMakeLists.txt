#
# @file
#

# The Freedesktop SDK the sandbox and the devcontainer are built on ships CMake
# 4.4: the upper bound of the range opts into every policy up to that release,
# while the lower bound keeps the tree configurable by an older CMake.
cmake_minimum_required(VERSION 3.30...4.4)

# No LANGUAGES here on purpose: the default list is what enables RC on Windows,
# which the resource files of the application and of the tests are compiled with.
project(squey)

###############################################################################
# Define testing environment
###############################################################################
# include(CTest) calls enable_testing() itself.
include(CTest)

###############################################################################
# Generic configuration for squey
###############################################################################

# Used for ICU, TBB and HWLoc
find_package(PkgConfig REQUIRED)

# Installation layout (CMAKE_INSTALL_LIBDIR, _BINDIR, _DATADIR...)
include(GNUInstallDirs)

# User defined macro
include(CMakeMacros.txt)

# Define option related to enabled/disabled features
include(CMakeOptions.txt)

# Define compilation option such CPP/CXXFLAGS, libs ...
include(CMakeCompilers.txt)

# windres does not understand the compiler flags CMake feeds a resource file by
# default (-fopenmp among them, which makes it bail out printing its usage), so
# reduce the command to what it actually needs. Set here rather than per
# directory: both the application and the test executables compile a .rc.
if (WIN32)
	set(CMAKE_RC_COMPILE_OBJECT "<CMAKE_RC_COMPILER> -O coff -F pe-x86-64 -o <OBJECT> <SOURCE>")
endif()

# Define folder for find_package macro
list(APPEND CMAKE_MODULE_PATH "${squey_SOURCE_DIR}/cmake")

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
include(CMakePackageDistribute.txt)

###############################################################################
# Add generic squey includes
###############################################################################

# PVBase. An interface target rather than a global include directory: whoever
# links it gets the headers, and nobody else does.
add_library(squey_base INTERFACE)
target_include_directories(squey_base INTERFACE "${squey_SOURCE_DIR}/include")
# pvbase/types.h builds its strong typedefs on type_safe, so whoever includes
# these headers needs it too.
target_link_libraries(squey_base INTERFACE squey_type_safe)

###############################################################################
# Define tests targets
###############################################################################
# Declare test suite target
# It has been populated thanks to the declare_pv_test macro

add_custom_target(squey_testsuite)
if (CMAKE_CONFIGURATION_TYPES)
    set(BUILD_CONF_OPTION "--build-config $<CONFIGURATION>")
endif()
add_custom_target(
    squey_run_testsuite COMMAND ${CMAKE_CTEST_COMMAND} -R SQUEY_TEST_
    --force-new-ctest-process --output-on-failure
    ${BUILD_CONF_OPTION}
)

add_dependencies(squey_testsuite squey)
add_dependencies(squey_run_testsuite squey_testsuite)
add_custom_target(squey_bench COMMAND ${CMAKE_CTEST_COMMAND} -R SQUEY_BENCH_)
add_dependencies(squey_bench squey_testsuite)

# ###############################################################################
# # furl
# ###############################################################################
add_compile_definitions(FURL_UTF8_CHAR)
add_subdirectory(furl)
# The submodules declare their targets without usage requirements; attach them
# here rather than patching repositories of their own. PUBLIC, not INTERFACE:
# the libraries are compiled against their own headers too.
target_include_directories(furls-utf8 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/furl/src/lib/include")

###############################################################################
# squey-utils
###############################################################################
add_subdirectory(squey-utils)
# The submodules link each other through these variables; they now name the
# targets instead of the file each library ends up at, so that CMake orders the
# builds and propagates the headers on its own.
set(PVLOGGER_LIBRARIES pvlogger)
set(PVHWLOC_LIBRARIES pvhwloc)
target_include_directories(pvlogger PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/squey-utils/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/squey-utils/libpvlogger/include")
target_include_directories(pvhwloc PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/squey-utils/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/squey-utils/libpvhwloc/include")

###############################################################################
# libpvcop
###############################################################################
add_subdirectory(libpvcop)
set(PVCOP_LIBRARIES pvcop)
target_include_directories(pvcop PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libpvcop/include")

squey_speedup_compilation(pvcop)

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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests" COMPONENT tests)
