# Locate PVHive include paths and libraries
#
# \file FindPVHive.cmake
#
# Copyright (C) Picviz Labs 2010-2012
#
# PVPARALLELVIEW_FOUND        - If PVHive is found
# PVPARALLELVIEW_INCLUDE_DIRS - Where include/pvparallelview is found
# PVPARALLELVIEW_LIBRARIES    - List of libraries when using pvparallelview
# PVPARALLELVIEW_DEFINITIONS  - List of definitions to be added when using pvparallelview
#

set(PVPARALLELVIEW_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVPARALLELVIEW_FOUND true)
		set(PVPARALLELVIEW_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}\\libparallelview\\src\\include")
		set(PVPARALLELVIEW_LIBRARY "${picviz-inspector_SOURCE_DIR}\\libparallelview\\src\\${CMAKE_BUILD_TYPE}\\pvparallelview.lib")
	else(WIN32)
		set(PVPARALLELVIEW_FOUND true)
		set(PVPARALLELVIEW_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}/libparallelview/src/include")
		set(PVPARALLELVIEW_LIBRARY "${picviz-inspector_SOURCE_DIR}/libparallelview/src/libpvparallelview.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVPARALLELVIEW_INCLUDE_DIR pvparallelview/PVHive.h
	          HINTS "../libparallelview/src/include" ${PVPARALLELVIEW_INCLUDEDIR}
	          PATH_SUFFIXES pvparallelview )

	find_library(PVPARALLELVIEW_LIBRARY NAMES pvparallelview
	             HINTS "../libparallelview/src/" ${PVPARALLELVIEW_LIBDIR} 
		     PATH_SUFFIXES libpvparallelview/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVPARALLELVIEW_LIBRARIES ${PVPARALLELVIEW_LIBRARY})
MESSAGE(STATUS "PVParallelView Libraries:" ${PVPARALLELVIEW_LIBRARIES})
set(PVPARALLELVIEW_INCLUDE_DIRS ${PVPARALLELVIEW_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVPARALLELVIEW  DEFAULT_MSG
                                  PVPARALLELVIEW_LIBRARIES PVPARALLELVIEW_INCLUDE_DIR)

mark_as_advanced(PVPARALLELVIEW_INCLUDE_DIRS PVPARALLELVIEW_LIBRARIES)

