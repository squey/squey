# Locate PVKernel include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
#
# PVKERNEL_FOUND        - If PVKernel is found
# PVKERNEL_INCLUDE_DIRS - Where include/pvkernel is found
# PVKERNEL_LIBRARIES    - List of libraries when using pvkernel
# PVKERNEL_DEFINITIONS  - List of definitions to be added when using pvkernel
#

set(PVKERNEL_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVKERNEL_FOUND true)
		set(PVKERNEL_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpvkernel\\src\\include")
		set(PVKERNEL_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpvkernel\\src\\${CMAKE_BUILD_TYPE}\\pvkernel.lib")
	else(WIN32)
		set(PVKERNEL_FOUND true)
		set(PVKERNEL_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpvkernel/src/include")
		set(PVKERNEL_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpvkernel/src/libpvkernel.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
	find_path(PVKERNEL_INCLUDE_DIR pvkernel/general.h
	          HINTS "../libpvkernel/src/include" ${PVKERNEL_INCLUDEDIR}
	          PATH_SUFFIXES pvkernel )

	find_library(PVKERNEL_LIBRARY NAMES pvkernel
	             HINTS "../libpvkernel/src/" ${PVKERNEL_LIBDIR} 
		     PATH_SUFFIXES libpvkernel/src/ )
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVKERNEL_LIBRARIES ${PVKERNEL_LIBRARY} ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})
set(PVKERNEL_INCLUDE_DIRS ${PVKERNEL_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVKERNEL  DEFAULT_MSG
                                  PVKERNEL_LIBRARIES PVKERNEL_INCLUDE_DIR)

mark_as_advanced(PVKERNEL_INCLUDE_DIRS PVKERNEL_LIBRARIES)

