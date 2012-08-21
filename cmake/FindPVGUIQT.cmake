# Locate PVHive include paths and libraries
#
# \file FindPVHive.cmake
#
# Copyright (C) Picviz Labs 2010-2012
#
# PVGUIQT_FOUND        - If PVGUIQT is found
# PVGUIQT_INCLUDE_DIRS - Where include/pvguiqt is found
# PVGUIQT_LIBRARIES    - List of libraries when using pvhive
# PVGUIQT_DEFINITIONS  - List of definitions to be added when using pvhive
#

set(PVGUIQT_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVGUIQT_FOUND true)
		set(PVGUIQT_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}\\libpvguiqt\\src\\include")
		set(PVGUIQT_LIBRARY "${picviz-inspector_SOURCE_DIR}\\libpvguiqt\\src\\${CMAKE_BUILD_TYPE}\\pvguiqt.lib")
	else(WIN32)
		set(PVGUIQT_FOUND true)
		set(PVGUIQT_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}/libpvguiqt/src/include")
		set(PVGUIQT_LIBRARY "${picviz-inspector_SOURCE_DIR}/libpvguiqt/src/libpvguiqt.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVGUIQT_INCLUDE_DIR pvhive/PVHive.h
	          HINTS "../libpvguiqt/src/include" ${PVGUIQT_INCLUDEDIR}
	          PATH_SUFFIXES pvGUIQT )

	find_library(PVGUIQT_LIBRARY NAMES pvhive
	             HINTS "../libpvguiqt/src/" ${PVGUIQT_LIBDIR} 
		     PATH_SUFFIXES libpvguiqt/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVGUIQT_LIBRARIES ${PVGUIQT_LIBRARY})
MESSAGE(STATUS "PVGUIQT Libraries:" ${PVGUIQT_LIBRARIES})
set(PVGUIQT_INCLUDE_DIRS ${PVGUIQT_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVGUIQT  DEFAULT_MSG
                                  PVGUIQT_LIBRARIES PVGUIQT_INCLUDE_DIR)

mark_as_advanced(PVGUIQT_INCLUDE_DIRS PVGUIQT_LIBRARIES)
