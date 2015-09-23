#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2015

project(libpvkernel)
include(CTest)
set(CTEST_PROJECT_NAME "PVKernel")

################################################
# QT5 related include and check
################################################

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(${Qt5Xml_INCLUDE_DIRS})

# For colors
find_package(Qt5Gui REQUIRED)
include_directories(${Qt5Gui_INCLUDE_DIRS})

# For Dialog
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

# FIXME : This is a hack waiting for Qt 5.4 and QOpenGLWidgets
set(QT_NO_OPENGL 1)
if(${QT_NO_OPENGL})
	add_definitions("-DQT_NO_OPENGL")
else(${QT_NO_OPENGL})
# For QGLWidget
find_package(Qt5OpenGL REQUIRED)
include_directories(${Qt5OpenGL_INCLUDE_DIRS})
endif(${QT_NO_OPENGL})

# For QtWebKit or QtWebEngine
if(${QT_WEBKIT})
	find_package(Qt5WebKitWidgets REQUIRED)
	include_directories(${Qt5WebKitWidgets_INCLUDE_DIRS})
else(${QT_WEBKIT})
	find_package(Qt5WebEngineWidgets REQUIRED)
	include_directories(${Qt5WebEngineWidgets_INCLUDE_DIRS})
endif(${QT_WEBKIT})

###################################################################
# Add include directories for picviz's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})
include_directories(AFTER ${ARCHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PCAP_INCLUDE_DIRS})
include_directories(AFTER ${X11_INCLUDE_DIR})
include_directories(AFTER ${Boost_INCLUDE_DIR})
include_directories(AFTER ${HWLOC_INCLUDE_DIRS})
if(CUDA_FOUND)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)
if(PYTHONLIBS_FOUND)
	include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
endif(PYTHONLIBS_FOUND)

###############################################################################
# Define plugins ressources
###############################################################################

# Set PVFILTER_NORMALIZE_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
        set(PVFILTER_NORMALIZE_DIR "normalize-filters")
else()
        set(PVFILTER_NORMALIZE_DIR "../../libpvkernel/plugins/normalize/")
endif()
add_definitions(-DPVFILTER_NORMALIZE_DIR="${PVFILTER_NORMALIZE_DIR}")

# Set PVRUSH_NORMALIZE_HELPERS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
        set(PVRUSH_NORMALIZE_HELPERS_DIR "normalize-helpers")
else()
	# This one is in the source dir, not the build dir
        set(PVRUSH_NORMALIZE_HELPERS_DIR "../../../libpvkernel/plugins/normalize-helpers")
endif()
add_definitions(-DPVRUSH_NORMALIZE_HELPERS_DIR="${PVRUSH_NORMALIZE_HELPERS_DIR}")

# Set PVRUSH_INPUTTYPE_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PVRUSH_INPUTTYPE_DIR "input-types")
else()
	set(PVRUSH_INPUTTYPE_DIR "../../libpvkernel/plugins/input_types/")
endif()
add_definitions(-DPVRUSH_INPUTTYPE_DIR="${PVRUSH_INPUTTYPE_DIR}")

# Set PVRUSH_SOURCE_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PVRUSH_SOURCE_DIR "sources")
else()
	set(PVRUSH_SOURCE_DIR "../../libpvkernel/plugins/sources/")
endif()
add_definitions(-DPVRUSH_SOURCE_DIR="${PVRUSH_SOURCE_DIR}")

# Set PICVIZ_QUERYBUILDER_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_QUERYBUILDER_DIR "querybuilder")
else()
	set(PICVIZ_QUERYBUILDER_DIR "../../../libpvkernel/src/widgets/querybuilder")
endif()
add_definitions(-DPICVIZ_QUERYBUILDER_DIR="${PICVIZ_QUERYBUILDER_DIR}")

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)

set(PVKERNEL_PLUGINS_LIST ${PLUGINS_LIST})

add_subdirectory(tests)
propagate_tests_targets_to_parent()
