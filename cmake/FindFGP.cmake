# - Try to find libfgp
# Once done this will define
#  FGP_FOUND - System has FGP
#  FGP_INCLUDE_DIRS - The FGP include directories
#  FGP_LIBRARIES - The libraries needed to use FGP

find_package(PkgConfig)

find_path(FGP_INCLUDE_DIR core/function.h
          PATH_SUFFIXES fgp)

find_library(FGP_LIBRARY NAMES fgp libfgp)

set(FGP_LIBRARIES ${FGP_LIBRARY} )
set(FGP_INCLUDE_DIRS ${FGP_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FGP_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(FGP  DEFAULT_MSG
                                  FGP_LIBRARY FGP_INCLUDE_DIR)

message(STATUS "FGP include dirs: ${FGP_INCLUDE_DIRS}")
message(STATUS "FGP libraries: ${FGP_LIBRARIES}")

mark_as_advanced(FGP_INCLUDE_DIR FGP_LIBRARY )
