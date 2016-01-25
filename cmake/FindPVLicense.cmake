# - Try to find libpvlicense
# Once done this will define
#  PVLICENSE_FOUND - System has PVLICENSE
#  PVLICENSE_INCLUDE_DIRS - The PVLICENSE include directories
#  PVLICENSE_LIBRARIES - The libraries needed to use PVLICENSE

find_package(PkgConfig)

find_path(PVLICENSE_INCLUDE_DIR esisw-flex.h
          PATH_SUFFIXES pvlicense HINTS /opt/pvlicense/include)

find_library(PVLICENSE_LIBRARY NAMES libesiflex.a HINTS /opt/pvlicense/lib)

set(PVLICENSE_LIBRARIES ${PVLICENSE_LIBRARY} dl)
set(PVLICENSE_INCLUDE_DIRS ${PVLICENSE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PVLICENSE_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVLICENSE  DEFAULT_MSG
                                  PVLICENSE_LIBRARIES PVLICENSE_INCLUDE_DIRS)

message(STATUS "PVLICENSE include dirs: ${PVLICENSE_INCLUDE_DIRS}")
message(STATUS "PVLICENSE libraries: ${PVLICENSE_LIBRARIES}")

mark_as_advanced(PVLICENSE_INCLUDE_DIR PVLICENSE_LIBRARY PVLOGGER_LIBRARY PVLOGGER_INCLUDE_DIR)
