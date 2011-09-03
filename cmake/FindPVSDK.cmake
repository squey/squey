# Locate PVSDK include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
#
# PVSDK_FOUND        - If Pvsdk is found
# PVSDK_INCLUDE_DIRS - Where include/pvsdk is found
# PVSDK_LIBRARIES    - List of libraries when using pvsdk
# PVSDK_DEFINITIONS  - List of definitions to be added when using pvsdk
#

set(PVSDK_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVSDK_FOUND true)
		set(PVSDK_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpvsdk\\src\\include")
		set(PVSDK_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpvsdk\\src\\${CMAKE_BUILD_TYPE}\\pvsdk.lib")
	else(WIN32)
		set(PVSDK_FOUND true)
		set(PVSDK_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpvsdk/src/include")
		set(PVSDK_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpvsdk/src/libpvsdk.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
if (WIN32)
	find_path(PVSDK_INCLUDE_DIR pvsdk\\general.h
	          HINTS "..\\libpvsdk\\src\\include" ${PVSDK_INCLUDEDIR}
	          PATH_SUFFIXES pvsdk )

	find_library(PVSDK_LIBRARY NAMES pvsdk
	             HINTS "..\\libpvsdk\\src\\Release" "libpvsdk\\src\\RelWithDebInfo" ${PVSDK_LIBDIR} 
		     PATH_SUFFIXES "libpvsdk\\src\\Release" "libpvsdk\\src\\RelWithDebInfo" )
else (WIN32)
	find_path(PVSDK_INCLUDE_DIR pvsdk/general.h
	          HINTS "../libpvsdk/src/include" ${PVSDK_INCLUDEDIR}
	          PATH_SUFFIXES pvsdk )

	find_library(PVSDK_LIBRARY NAMES pvsdk
	             HINTS "../libpvsdk/src/" ${PVSDK_LIBDIR} 
		     PATH_SUFFIXES libpvsdk/src/ )
endif(WIN32)
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVSDK_INCLUDE_DIRS ${PVSDK_INCLUDE_DIR})
set(PVSDK_LIBRARIES ${PVSDK_LIBRARY})

set(PVSDK_DEFINITIONS ${PVSDK_DEFINITIONS})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVSDK  DEFAULT_MSG
                                  PVSDK_LIBRARY PVSDK_INCLUDE_DIR)

mark_as_advanced(PVSDK_INCLUDE_DIRS PVSDK_LIBRARIES)
