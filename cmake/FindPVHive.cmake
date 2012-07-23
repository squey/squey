# Locate PVHive include paths and libraries
#
# \file FindPVHive.cmake
#
# Copyright (C) Picviz Labs 2010-2012
#
# PVHIVE_FOUND        - If PVHive is found
# PVHIVE_INCLUDE_DIRS - Where include/pvhive is found
# PVHIVE_LIBRARIES    - List of libraries when using pvhive
# PVHIVE_DEFINITIONS  - List of definitions to be added when using pvhive
#

set(PVHIVE_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVHIVE_FOUND true)
		set(PVHIVE_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}\\libpvhive\\src\\include")
		set(PVHIVE_LIBRARY "${picviz-inspector_SOURCE_DIR}\\libpvhive\\src\\${CMAKE_BUILD_TYPE}\\pvhive.lib")
	else(WIN32)
		set(PVHIVE_FOUND true)
		set(PVHIVE_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}/libpvhive/src/include")
		set(PVHIVE_LIBRARY "${picviz-inspector_SOURCE_DIR}/libpvhive/src/libpvhive.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVHIVE_INCLUDE_DIR pvhive/PVHive.h
	          HINTS "../libpvhive/src/include" ${PVHIVE_INCLUDEDIR}
	          PATH_SUFFIXES pvhive )

	find_library(PVHIVE_LIBRARY NAMES pvhive
	             HINTS "../libpvhive/src/" ${PVHIVE_LIBDIR} 
		     PATH_SUFFIXES libpvhive/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVHIVE_LIBRARIES ${PVHIVE_LIBRARY})
MESSAGE(STATUS "PVHive Libraries:" ${PVHIVE_LIBRARIES})
set(PVHIVE_INCLUDE_DIRS ${PVHIVE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVHIVE  DEFAULT_MSG
                                  PVHIVE_LIBRARIES PVHIVE_INCLUDE_DIR)

mark_as_advanced(PVHIVE_INCLUDE_DIRS PVHIVE_LIBRARIES)

