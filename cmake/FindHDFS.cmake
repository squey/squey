###################################################################
# - Find HDFS
# Find the HDFS includes and library
# http://hadoop.apache.org
#
# It also looks for libjni (because libhdfs needs it).
#
# The environment variable HDFSDIR allows to specficy where to find 
# libhdfs in non standard location.
#  
#  HDFS_INCLUDE_DIRS - where to find hdfs.h
#  HDFS_LIBRARIES   - List of libraries when using libhdfs
#  HDFS_FOUND       - True if libhdfs found.


IF(EXISTS $ENV{HDFSDIR})
  FIND_PATH(HDFS_INCLUDE_DIR 
    NAMES
    hadoop/hdfs.h
    hdfs.h
    PATHS
      $ENV{HDFSDIR}
    NO_DEFAULT_PATH

  )
  
  FIND_LIBRARY(HDFS_LIBRARY
    NAMES 
      hdfs
    PATHS
      $ENV{HDFSDIR}
    NO_DEFAULT_PATH
  )
  

ELSE(EXISTS $ENV{HDFSDIR})
  FIND_PATH(HDFS_INCLUDE_DIR 
    NAMES
    hadoop/hdfs.h
    hdfs.h
  )
  
  FIND_LIBRARY(HDFS_LIBRARY
    NAMES 
      hdfs
  )
  
ENDIF(EXISTS $ENV{HDFSDIR})

# Search for libjni
FIND_PACKAGE(JNI)

# If it is found, then we can go on
IF (JNI_FOUND)
	IF (HDFS_INCLUDE_DIR)
		IF (HDFS_LIBRARY)
			# TOFIX: here JNI_LIBRARIES isn't used because it includes
			# AWT, and it seems link libmawt is missing from the libraries
			# we should be linking with. So we only use ${JAVA_JVM_LIBRARY}.
			# (see FindJNI.cmake in the cmake tree for more information)
			SET(HDFS_LIBRARIES ${HDFS_LIBRARY} ${JAVA_JVM_LIBRARY})
			SET(HDFS_INCLUDE_DIRS ${HDFS_INCLUDE_DIR} ${JNI_INCLUDE_DIRS})

			MESSAGE(STATUS "HDFS include dirs set to ${HDFS_INCLUDE_DIRS}")
			MESSAGE(STATUS "HDFS libraries set to  ${HDFS_LIBRARIES}")

			INCLUDE(FindPackageHandleStandardArgs)
			FIND_PACKAGE_HANDLE_STANDARD_ARGS(HDFS DEFAULT_MSG HDFS_INCLUDE_DIRS HDFS_LIBRARIES)

			MARK_AS_ADVANCED(
				HDFS_LIBRARIES
				HDFS_INCLUDE_DIRS
				)
		ELSE (HDFS_LIBRARY)
			MESSAGE(WARNING "libhdfs library can't be found. HDFS support is disabled.")
		ENDIF(HDFS_LIBRARY)
	ELSE(HDFS_INCLUDE_DIR)
		MESSAGE(WARNING "libhdfs include directory can't be found. HDFS support is disabled")
	ENDIF(HDFS_INCLUDE_DIR)
ELSE(JNI_FOUND)
	MESSAGE(WARNING "JNI library can't be found. HDFS support is disabled")
ENDIF(JNI_FOUND)
