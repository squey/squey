# - Try to find libpvcop
# Once done this will define
#  PVCOP_FOUND - System has PVCOP
#  PVCOP_INCLUDE_DIRS - The PVCOP include directories
#  PVCOP_LIBRARIES - The libraries needed to use PVCOP

find_package(PkgConfig)

find_path(PVCOP_INCLUDE_DIR pvcop.h
          PATH_SUFFIXES pvcop)

find_library(PVCOP_LIBRARY NAMES pvcop libpvcop)
find_library(PVLOGGER_LIBRARY NAMES pvlogger libpvlogger)

set(PVCOP_LIBRARIES ${PVCOP_LIBRARY} ${PVLOGGER_LIBRARY})
set(PVCOP_INCLUDE_DIRS ${PVCOP_INCLUDE_DIR} ${PVLOGGER_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PVCOP_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVCOP  DEFAULT_MSG
                                  PVCOP_LIBRARIES PVCOP_INCLUDE_DIRS)

message(STATUS "PVCOP include dirs: ${PVCOP_INCLUDE_DIRS}")
message(STATUS "PVCOP libraries: ${PVCOP_LIBRARIES}")

mark_as_advanced(PVCOP_INCLUDE_DIR PVCOP_LIBRARY PVLOGGER_LIBRARY PVLOGGER_INCLUDE_DIR)
