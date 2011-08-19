# Locate Picviz include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
#
# PICVIZ_FOUND        - If Picviz is found
# PICVIZ_INCLUDE_DIRS - Where include/picviz is found
# PICVIZ_LIBRARIES    - List of libraries when using picviz
# PICVIZ_DEFINITIONS  - List of definitions to be added when using picviz
#

set(PICVIZ_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PICVIZ_FOUND true)
		set(PICVIZ_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpicviz\\src\\include")
		set(PICVIZ_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpicviz\\src\\${CMAKE_BUILD_TYPE}\\picviz.lib")
	else(WIN32)
		set(PICVIZ_FOUND true)
		set(PICVIZ_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include")
		set(PICVIZ_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/libpicviz.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
if (WIN32)
	find_path(PICVIZ_INCLUDE_DIR picviz\\general.h
	          HINTS "..\\libpicviz\\src\\include" ${PICVIZ_INCLUDEDIR}
	          PATH_SUFFIXES picviz )

	find_library(PICVIZ_LIBRARY NAMES picviz
	             HINTS "..\\libpicviz\\src\\Release" "libpicviz\\src\\RelWithDebInfo" ${PICVIZ_LIBDIR} 
		     PATH_SUFFIXES "libpicviz\\src\\Release" "libpicviz\\src\\RelWithDebInfo" )
else (WIN32)
	find_path(PICVIZ_INCLUDE_DIR picviz/general.h
	          HINTS "../libpicviz/src/include" ${PICVIZ_INCLUDEDIR}
	          PATH_SUFFIXES picviz )

	find_library(PICVIZ_LIBRARY NAMES picviz
	             HINTS "../libpicviz/src/" ${PICVIZ_LIBDIR} 
		     PATH_SUFFIXES libpicviz/src/ )
endif(WIN32)
endif(PICVIZ_SINGLE_TREE_BUILD)

#
# Check for PCAP library
#
if(NOT WIN32)
	find_package(PCAP REQUIRED)
else(NOT WIN32)
	set(PCAP_LIBRARIES "C:\\dev\\winpcap\\Lib\\wpcap.lib")
	set(PCAP_INCLUDE_DIRS "C:\\dev\\winpcap\\Include")
endif(NOT WIN32)

#
# PCRE
#
# if(WIN32)
# 	set(PCRE_INCLUDE_DIR "C:\\dev\\GnuWin32\\include")
#         set(PCRE_PCREPOSIX_LIBRARY "C:\\dev\\GnuWin32\\lib\\pcreposix.lib")
#         set(PCRE_PCRE_LIBRARY "C:\\dev\\GnuWin32\\lib\\pcre.lib")
# else(WIN32)
# 	FIND_PACKAGE(PCRE REQUIRED)
# endif(WIN32)

#
# WinLicense
# 
#if(WIN32)
#	set(WINLICENSE_LIBRARY "C:\\dev\\WinlicenseSDK.lib")
#	add_definitions(-DWIN32_LEAN_AND_MEAN)
#endif(WIN32)

# Find Perl
if(WIN32)
	set(PERLLIBS_FOUND "FALSE")
else(WIN32)
	FIND_PACKAGE(PerlLibs)
endif(WIN32)


set(PICVIZ_INCLUDE_DIRS ${PICVIZ_INCLUDE_DIR} ${PCAP_INCLUDE_DIRS})
set(PICVIZ_LIBRARIES ${PICVIZ_LIBRARY} ${PCAP_LIBRARIES} ${WINLICENSE_LIBRARY})
set(PICVIZ_LIBRARIES_WITHOUT_PICVIZ ${PCAP_LIBRARIES} ${WINLICENSE_LIBRARY})

set(PICVIZ_DEFINITIONS ${PICVIZ_DEFINITIONS})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PICVIZ  DEFAULT_MSG
                                  PICVIZ_LIBRARY PICVIZ_INCLUDE_DIR)

mark_as_advanced(PICVIZ_INCLUDE_DIRS PICVIZ_LIBRARIES PICVIZ_INCLUDES_WITHOUT_PICVIZ)
