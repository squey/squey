# Locate Pvgl include paths and libraries
#
# \file FindPVGL2.cmake
#
# Copyright (C) Picviz Labs 2010-2012
#
# PVGL_FOUND        - If Pvgl is found
# PVGL_INCLUDE_DIRS - Where include/pvgl is found
# PVGL_LIBRARIES    - List of libraries when using pvgl
# PVGL_DEFINITIONS  - List of definitions to be added when using pvgl
#

set(PVGL_DEFINITIONS "")

if(PICVIZ_SINGLE_TREE_BUILD)
	if(WIN32)
		set(PVGL_FOUND true)
		set(PVGL_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}\\libpvgl2\\src\\include")
		set(PVGL_LIBRARY "${picviz-inspector_SOURCE_DIR}\\libpvgl2\\src\\${CMAKE_BUILD_TYPE}\\pvgl.lib")
	else(WIN32)
		set(PVGL_FOUND true)
		set(PVGL_INCLUDE_DIR "${picviz-inspector_SOURCE_DIR}/libpvgl2/src/include")
		set(PVGL_LIBRARY "${picviz-inspector_SOURCE_DIR}/libpvgl2/src/libpvgl.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
if (WIN32)
	find_path(PVGL_INCLUDE_DIR pvgl\\general.h
	          HINTS "..\\libpvgl2\\src\\include" ${PVGL_INCLUDEDIR}
	          PATH_SUFFIXES pvgl )

	find_library(PVGL_LIBRARY NAMES pvgl
	             HINTS "..\\libpvgl2\\src\\Release" "libpvgl2\\src\\RelWithDebInfo" ${PVGL_LIBDIR} 
		     PATH_SUFFIXES "libpvgl2\\src\\Release" "libpvgl2\\src\\RelWithDebInfo" )
else (WIN32)
	find_path(PVGL_INCLUDE_DIR pvgl/general.h
	          HINTS "../libpvgl2/src/include" ${PVGL_INCLUDEDIR}
	          PATH_SUFFIXES pvgl )

	find_library(PVGL_LIBRARY NAMES pvgl
	             HINTS "../libpvgl2/src/" ${PVGL_LIBDIR} 
		     PATH_SUFFIXES libpvgl/src/ )
endif(WIN32)
endif(PICVIZ_SINGLE_TREE_BUILD)

set(PVGL_INCLUDE_DIRS ${PVGL_INCLUDE_DIR} ${GLEW_INCLUDE_PATH} ${GLUT_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS})
set(PVGL_LIBRARIES ${PVGL_LIBRARY} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARIES})

set(PVGL_DEFINITIONS ${PVGL_DEFINITIONS} ${PVCORE_DEFINITIONS})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVGL  DEFAULT_MSG
                                  PVGL_LIBRARY PVGL_INCLUDE_DIR)

mark_as_advanced(PVGL_INCLUDE_DIRS PVGL_LIBRARIES)
