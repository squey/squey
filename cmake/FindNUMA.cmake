# Find the numa policy library.
# Output variables:
#  NUMA_INCLUDE_DIR : e.g., /usr/include/.
#  NUMA_LIBRARY     : Library path of numa library
#  NUMA_FOUND       : True if found.
FIND_PATH(NUMA_INCLUDE_DIR NAME numa.h
    HINTS /app/include $ENV{HOME}/local/include /opt/local/include /usr/local/include /usr/include)

FIND_LIBRARY(NUMA_LIBRARY NAME numa
    HINTS /app/lib $ENV{HOME}/local/lib64 $ENV{HOME}/local/lib /usr/local/lib64 /usr/local/lib /opt/local/lib64 /opt/local/lib /usr/lib64 /usr/lib
    )

IF (NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
    SET(NUMA_FOUND TRUE)
ELSE ()
    SET(NUMA_FOUND FALSE)
    MESSAGE(STATUS "WARNING: Numa library not found.")
ENDIF ()

SET(NUMA_LIBRARIES ${NUMA_LIBRARY})
