# Locate PVCore include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
#
# PVCORE_FOUND        - If PVCore is found
# PVCORE_INCLUDE_DIRS - Where include/pvcore is found
# PVCORE_LIBRARIES    - List of libraries when using pvcore
# PVCORE_DEFINITIONS  - List of definitions to be added when using pvcore
#

set(PVCORE_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVCORE_FOUND true)
		set(PVCORE_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpvcore\\src\\include")
		set(PVCORE_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpvcore\\src\\${CMAKE_BUILD_TYPE}\\pvcore.lib")
	else(WIN32)
		set(PVCORE_FOUND true)
		set(PVCORE_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include")
		set(PVCORE_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/libpvcore.so" -lgomp)
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVCORE_INCLUDE_DIR pvcore/general.h
	          HINTS "../libpvcore/src/include" ${PVCORE_INCLUDEDIR}
	          PATH_SUFFIXES pvcore )

	find_library(PVCORE_LIBRARY NAMES pvcore
	             HINTS "../libpvcore/src/" ${PVCORE_LIBDIR} 
		     PATH_SUFFIXES libpvcore/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVCORE_LIBRARIES ${PVCORE_LIBRARY})
set(PVCORE_INCLUDE_DIRS ${PVCORE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVCORE  DEFAULT_MSG
                                  PVCORE_LIBRARY PVCORE_INCLUDE_DIR)

mark_as_advanced(PVCORE_INCLUDE_DIRS PVCORE_LIBRARIES)

