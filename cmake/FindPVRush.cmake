# Locate Pvrush include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
#
# PVRUSH_FOUND        - If Pvrush is found
# PVRUSH_INCLUDE_DIRS - Where include/pvrush is found
# PVRUSH_LIBRARIES    - List of libraries when using pvrush
# PVRUSH_DEFINITIONS  - List of definitions to be added when using pvrush
#

set(PVRUSH_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVRUSH_FOUND true)
		set(PVRUSH_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpvrush\\src\\include")
		set(PVRUSH_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpvrush\\src\\${CMAKE_BUILD_TYPE}\\pvrush.lib")
	else(WIN32)
		set(PVRUSH_FOUND true)
		set(PVRUSH_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/include")
		set(PVRUSH_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/libpvrush.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVRUSH_INCLUDE_DIR pvrush/general.h
        	  HINTS "../libpvrush/src/include" ${PVRUSH_INCLUDEDIR}
	          PATH_SUFFIXES pvrush )

	find_library(PVRUSH_LIBRARY NAMES pvrush
        	     HINTS "../libpvrush/src/" ${PVRUSH_LIBDIR} 
		     PATH_SUFFIXES libpvrush/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVRUSH_LIBRARIES ${PVRUSH_LIBRARY})
set(PVRUSH_INCLUDE_DIRS ${PVRUSH_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVRUSH  DEFAULT_MSG
                                  PVRUSH_LIBRARY PVRUSH_INCLUDE_DIR)

mark_as_advanced(PVRUSH_INCLUDE_DIRS PVRUSH_LIBRARIES)

