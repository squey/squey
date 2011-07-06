cmake_minimum_required(VERSION 2.8)

project(vrb)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(CheckIncludeFiles)
include(CheckFunctionExists)
include(CheckLibraryExists)

include(FindPkgConfig)

include_directories(AFTER "${vrb_SOURCE_DIR}/include")

add_subdirectory(vrb)
