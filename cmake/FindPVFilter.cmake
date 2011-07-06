# Locate PVFilter include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
#
# PVFILTER_FOUND        - If Pvfilter is found
# PVFILTER_INCLUDE_DIRS - Where include/pvfilter is found
# PVFILTER_LIBRARIES    - List of libraries when using pvfilter
# PVFILTER_DEFINITIONS  - List of definitions to be added when using pvfilter
#

set(PVFILTER_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVFILTER_FOUND true)
		set(PVFILTER_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpvfilter\\src\\include")
		set(PVFILTER_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpvfilter\\src\\${CMAKE_BUILD_TYPE}\\pvfilter.lib")
	else(WIN32)
		set(PVFILTER_FOUND true)
		set(PVFILTER_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include")
		set(PVFILTER_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/libpvfilter.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVFILTER_INCLUDE_DIR pvfilter/general.h
        	  HINTS "../libpvfilter/src/include" ${PVFILTER_INCLUDEDIR}
	          PATH_SUFFIXES pvfilter )

	find_library(PVFILTER_LIBRARY NAMES pvfilter
        	     HINTS "../libpvfilter/src/" ${PVFILTER_LIBDIR} 
		     PATH_SUFFIXES libpvfilter/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVFILTER_LIBRARIES ${PVFILTER_LIBRARY})
set(PVFILTER_INCLUDE_DIRS ${PVFILTER_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVFILTER  DEFAULT_MSG
                                  PVFILTER_LIBRARY PVFILTER_INCLUDE_DIR)

mark_as_advanced(PVFILTER_INCLUDE_DIRS PVFILTER_LIBRARIES)

