#
# \file FindPICVIZ.cmake
#
# Copyright (C) Picviz Labs 2010-2012

#
# $Id$
#
#  PICVIZ_INCLUDE_DIRS - Where libpicviz headers are located.
#  PICVIZ_LIBRARIES    - libpicviz libraries list.
#  PICVIZ_FLAGS        - various flags (use add_definitions(PICVIZ_FLAGS)
#  PICVIZ_FOUND        - True if libpicviz found.


if(WIN32)
	FIND_PATH(PICVIZ_INCLUDE_DIR 
	    NAMES
	    picviz\\general.h
	    PATHS
	      "..\\libpicviz\\src\\include\\"
	    NO_DEFAULT_PATH 
	)

	FIND_LIBRARY(PICVIZ_LIBRARY
	    NAMES 
	      picviz
	    PATHS
	    #"..\\libpicviz\\src\\Release\\"
	    "..\\libpicviz\\src\\RelWithDebInfo\\"
	    #	      "..\\libpicviz\\src\\Debug\\"
	    NO_DEFAULT_PATH 
	)
	
#	set(PICVIZ_LIBRARY "D:\\cactuslabs\\trunk\\libpicviz\\src\\Debug\\picviz.lib")

else(WIN32)
	FIND_PATH(PICVIZ_INCLUDE_DIR 
	    NAMES
	    picviz/general.h
	    PATHS
	      "../libpicviz/src/include/"
	    NO_DEFAULT_PATH 
	)

       # add_library(picviz STATIC IMPORTED)
       # set_property(TARGET picviz PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../libpicviz/src/libpicviz.a)
       # set(PICVIZ_LIBRARY picviz m)
  
	FIND_LIBRARY(PICVIZ_LIBRARY
	    NAMES 
	      picviz
	    PATHS
	      "../libpicviz/src/"
	    NO_DEFAULT_PATH 
	)
endif(WIN32)

#
# Check for APR (Apache Portable Runtime)
#
if(NOT WIN32)
       find_package(APR REQUIRED)
else(NOT WIN32)
	#ws2_32.lib and mswsock.lib
       set(APR_INCLUDES "C:\\dev\\apr\\include")
       set(APR_FLAGS -DAPR_DECLARE_STATIC)
       set(APR_LIBS "C:\\dev\\apr\\LibD\\apr-1.lib" "ws2_32")
endif(NOT WIN32)

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
# Check for DUMBNET library
#
if(NOT WIN32)
	find_package(DUMBNET REQUIRED)
else(NOT WIN32)
	set(DUMBNET_LIBRARIES "C:\\dev\\libdnet\\lib\\dnet.lib")
	set(DUMBNET_INCLUDE_DIRS "C:\\dev\\libdnet\\include")
endif(NOT WIN32)

#
# PCRE
#
if(WIN32)
	set(PCRE_INCLUDE_DIR "C:\\dev\\GnuWin32\\include")
        set(PCRE_PCREPOSIX_LIBRARY "C:\\dev\\GnuWin32\\lib\\pcreposix.lib")
        set(PCRE_PCRE_LIBRARY "C:\\dev\\GnuWin32\\lib\\pcre.lib")
else(WIN32)
	FIND_PACKAGE(PCRE REQUIRED)
endif(WIN32)

#
# WinLicense
# 
if(WIN32)
	set(WINLICENSE_LIBRARY "C:\\dev\\WinlicenseSDK.lib")
	add_definitions(-DWIN32_LEAN_AND_MEAN)
endif(WIN32)


SET(PICVIZ_INCLUDE_DIRS ${PICVIZ_INCLUDE_DIR} ${PCRE_INCLUDE_DIR} ${DUMBNET_INCLUDE_DIRS} ${PCAP_INCLUDE_DIRS} ${APR_INCLUDES})
SET(PICVIZ_LIBRARIES ${PICVIZ_LIBRARY} ${PCRE_PCREPOSIX_LIBRARY} ${PCRE_PCRE_LIBRARY} ${DUMBNET_LIBRARIES} ${PCAP_LIBRARIES} ${APR_LIBS} ${WINLICENSE_LIBRARY})

SET(PICVIZ_FLAGS ${APR_FLAGS})

IF(PICVIZ_INCLUDE_DIRS)
  MESSAGE(STATUS "Picviz include dirs set to ${PCAP_INCLUDE_DIRS}")
ELSE(PICVIZ_INCLUDE_DIRS)
  MESSAGE(FATAL " Picviz include dirs cannot be found")
ENDIF(PICVIZ_INCLUDE_DIRS)

IF(PICVIZ_LIBRARIES)
  MESSAGE(STATUS "Picviz library set to  ${PICVIZ_LIBRARIES}")
ELSE(PICVIZ_LIBRARIES)
  MESSAGE(FATAL "Picviz library cannot be found")
ENDIF(PICVIZ_LIBRARIES)

MARK_AS_ADVANCED(
  PICVIZ_LIBRARIES
  PICVIZ_INCLUDE_DIRS
)
