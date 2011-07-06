# Locate Pvgl include paths and libraries
#
# $Id$
# Copyright (C) Sebastien Tricaud 2010-2011
# Copyright (C) Philippe Saade 2010-2011
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
		set(PVGL_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}\\libpvgl\\src\\include")
		set(PVGL_LIBRARY "${Picviz_Inspector_SOURCE_DIR}\\libpvgl\\src\\${CMAKE_BUILD_TYPE}\\pvgl.lib")
	else(WIN32)
		set(PVGL_FOUND true)
		set(PVGL_INCLUDE_DIR "${Picviz_Inspector_SOURCE_DIR}/libpvgl/src/include")
		set(PVGL_LIBRARY "${Picviz_Inspector_SOURCE_DIR}/libpvgl/src/libpvgl.so")
	endif(WIN32)
else(PICVIZ_SINGLE_TREE_BUILD)
if (WIN32)
	find_path(PVGL_INCLUDE_DIR pvgl\\general.h
	          HINTS "..\\libpvgl\\src\\include" ${PVGL_INCLUDEDIR}
	          PATH_SUFFIXES pvgl )

	find_library(PVGL_LIBRARY NAMES pvgl
	             HINTS "..\\libpvgl\\src\\Release" "libpvgl\\src\\RelWithDebInfo" ${PVGL_LIBDIR} 
		     PATH_SUFFIXES "libpvgl\\src\\Release" "libpvgl\\src\\RelWithDebInfo" )
else (WIN32)
	find_path(PVGL_INCLUDE_DIR pvgl/general.h
	          HINTS "../libpvgl/src/include" ${PVGL_INCLUDEDIR}
	          PATH_SUFFIXES pvgl )

	find_library(PVGL_LIBRARY NAMES pvgl
	             HINTS "../libpvgl/src/" ${PVGL_LIBDIR} 
		     PATH_SUFFIXES libpvgl/src/ )
endif(WIN32)
endif(PICVIZ_SINGLE_TREE_BUILD)

#
# PVCore
#
find_package(PVCore REQUIRED)

#
# OpenGL
#
find_package(OpenGL REQUIRED)

#
# GLEW
#
if (WIN32)
	set(GLEW_INCLUDE_PATH "C:\\dev\\glew\\include")
	set(GLEW_LIBRARY "C:\\dev\\glew\\lib\\glew32s.lib")
else(WIN32)
	find_package(GLEW REQUIRED)
endif(WIN32)

#
# GLUT
#
if (WIN32)
	set(GLUT_INCLUDE_DIR "C:\\dev\\freeglut\\include\\")
	set(GLUT_glut_LIBRARY "C:\\dev\\freeglut\\freeglut.lib")
else(WIN32)
  find_package(GLUT REQUIRED)
endif(WIN32)
	
#
# Freetype
#
if (WIN32)
	set(FREETYPE_LIBRARY "C:\\dev\\GnuWin32\\lib\\freetype.lib")
	set(FREETYPE_INCLUDE_DIRS "C:\\dev\\GnuWin32\\include\\freetype2")
else(WIN32)
	find_package(Freetype REQUIRED)
endif(WIN32)


set(PVGL_INCLUDE_DIRS ${PVGL_INCLUDE_DIR} ${GLEW_INCLUDE_PATH} ${GLUT_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS})
set(PVGL_LIBRARIES ${PVGL_LIBRARY} ${GLEW_LIBRARY} ${GLUT_glut_LIBRARY} ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARIES})

set(PVGL_DEFINITIONS ${PVGL_DEFINITIONS} ${PVCORE_DEFINITIONS})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PVGL  DEFAULT_MSG
                                  PVGL_LIBRARY PVGL_INCLUDE_DIR)

mark_as_advanced(PVGL_INCLUDE_DIRS PVGL_LIBRARIES)
