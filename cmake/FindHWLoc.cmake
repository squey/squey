# - Try to find HWLoc
# Once done this will define
#  HWLOC_FOUND - System has HWLoc
#  HWLOC_INCLUDE_DIRS - The HWLoc include directories
#  HWLOC_LIBRARIES - The libraries needed to use HWLoc
#  HWLOC_DEFINITIONS - Compiler switches required for using HWLoc

find_package(PkgConfig)
pkg_check_modules(PC_HWLOC QUIET hwloc)
set(HWLOC_DEFINITIONS ${PC_HWLOC_CFLAGS_OTHER})

find_path(HWLOC_INCLUDE_DIR hwloc.h
          HINTS ${PC_HWLOCL_INCLUDEDIR} ${PC_HWLOC_INCLUDE_DIRS}
          PATH_SUFFIXES hwloc)

find_library(HWLOC_LIBRARY NAMES hwloc libhwloc
             HINTS ${PC_HWLOC_LIBDIR} ${PC_HWLOC_LIBRARY_DIRS} )

set(HWLOC_LIBRARIES ${HWLOC_LIBRARY} )
set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set HWLOC_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(HWloc  DEFAULT_MSG
                                  HWLOC_LIBRARY HWLOC_INCLUDE_DIR)

mark_as_advanced(HWLOC_INCLUDE_DIR HWLOC_LIBRARY )