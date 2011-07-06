###################################################################
# - Find the Dumb (not so!) Library: dnet
# Find the DUMBNET includes and library
# http://code.google.com/p/libdnet/
#
# The environment variable DUMBNETDIR allows to specficy where to find 
# libdnet in non standard location.
#  
#  DUMBNET_INCLUDE_DIRS - where to find dnet.h, etc.
#  DUMBNET_LIBRARIES   - List of libraries when using dnet.
#  DUMBNET_FOUND       - True if dnet found.


IF(EXISTS $ENV{DUMBNETDIR})
  FIND_PATH(DUMBNET_INCLUDE_DIR 
    NAMES
    dnet.h
    dnet/ip.h
    dnet/tcp.h
    dnet/udp.h
    PATHS
      $ENV{DUMBNETDIR}
    NO_DEFAULT_PATH
  )
  
  FIND_LIBRARY(DUMBNET_LIBRARY
    NAMES 
      dnet
    PATHS
      $ENV{DUMBNETDIR}
    NO_DEFAULT_PATH
  )

# Because DEBIAN _is_ specific :(  
  FIND_PATH(DUMBNET_INCLUDE_DIR 
    NAMES
    dumbnet.h
    dumbnet/ip.h
    dumbnet/tcp.h
    dumbnet/udp.h
    PATHS
      $ENV{DUMBNETDIR}
    NO_DEFAULT_PATH
  )
  IF(EXISTS $ENV{DUMBNET_INCLUDE_DIR}/dumbnet.h)
    SET(OS_DEBIAN "YES")
  ENDIF(EXISTS $ENV{DUMBNET_INCLUDE_DIR}/dumbnet.h)
  
  FIND_LIBRARY(DUMBNET_LIBRARY
    NAMES 
      dumbnet
    PATHS
      $ENV{DUMBNETDIR}
    NO_DEFAULT_PATH
  )

ELSE(EXISTS $ENV{DUMBNETDIR})
  FIND_PATH(DUMBNET_INCLUDE_DIR 
    NAMES
    dnet.h
    dnet/ip.h
    dnet/tcp.h
    dnet/udp.h
  )
  
  FIND_LIBRARY(DUMBNET_LIBRARY
    NAMES 
      dnet
  )

# Because DEBIAN _is_ specific :(  
  FIND_PATH(DUMBNET_INCLUDE_DIR 
    NAMES
    dumbnet.h
    dumbnet/ip.h
    dumbnet/tcp.h
    dumbnet/udp.h
  )
  IF(EXISTS $ENV{DUMBNET_INCLUDE_DIR}/dumbnet.h)
    SET(OS_DEBIAN "YES")
  ENDIF(EXISTS $ENV{DUMBNET_INCLUDE_DIR}/dumbnet.h)
  
  FIND_LIBRARY(DUMBNET_LIBRARY
    NAMES 
      dumbnet
  )
  
ENDIF(EXISTS $ENV{DUMBNETDIR})

SET(DUMBNET_INCLUDE_DIRS ${DUMBNET_INCLUDE_DIR})
SET(DUMBNET_LIBRARIES ${DUMBNET_LIBRARY})

IF(DUMBNET_INCLUDE_DIRS)
  MESSAGE(STATUS "dnet include dirs set to ${DUMBNET_INCLUDE_DIRS}")
ELSE(DUMBNET_INCLUDE_DIRS)
  MESSAGE(FATAL " dnet include dirs cannot be found")
ENDIF(DUMBNET_INCLUDE_DIRS)

IF(DUMBNET_LIBRARIES)
  MESSAGE(STATUS "dnet library set to  ${DUMBNET_LIBRARIES}")
ELSE(DUMBNET_LIBRARIES)
  MESSAGE(FATAL "dnet library cannot be found")
ENDIF(DUMBNET_LIBRARIES)

#Functions
INCLUDE(CheckFunctionExists)
SET(CMAKE_REQUIRED_INCLUDES ${DUMBNET_INCLUDE_DIRS})
SET(CMAKE_REQUIRED_LIBRARIES ${DUMBNET_LIBRARIES})
CHECK_FUNCTION_EXISTS("ip_checksum" HAVE_DUMBNET_IPCHECKSUM)
CHECK_FUNCTION_EXISTS("ip_ntoa" HAVE_DUMBNET_IP_NTOA)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DUMBNET DEFAULT_MSG DUMBNET_INCLUDE_DIRS DUMBNET_LIBRARIES)

MARK_AS_ADVANCED(
  DUMBNET_LIBRARIES
  DUMBNET_INCLUDE_DIRS
)
