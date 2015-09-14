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

# For QGLWidget
find_package(Qt5OpenGL REQUIRED)
include_directories(${Qt5OpenGL_INCLUDE_DIRS})

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

##########################################
# begin: SET VARIOUS PLUGINS DIRECTORIES #
##########################################

# Set PVFILTER_NORMALIZE_DIR
if(WIN32)
        set(PVFILTER_NORMALIZE_DIR "normalize-filters")
else(WIN32)
        set(PVFILTER_NORMALIZE_DIR "${CMAKE_INSTALL_PREFIX}/share/pvfilter/normalize-filters/")
endif(WIN32)
add_definitions(-DPVFILTER_NORMALIZE_DIR="${PVFILTER_NORMALIZE_DIR}")

# Set PVRUSH_NORMALIZE_DIR
if(WIN32)
        set(PVRUSH_NORMALIZE_DIR "normalize")
else(WIN32)
        set(PVRUSH_NORMALIZE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/normalize/")
endif(WIN32)
add_definitions(-DPVRUSH_NORMALIZE_DIR="${PVRUSH_NORMALIZE_DIR}")

# Set PVRUSH_NORMALIZE_HELPERS_DIR
if(WIN32)
        set(PVRUSH_NORMALIZE_HELPERS_DIR "normalize-helpers")
else(WIN32)
        set(PVRUSH_NORMALIZE_HELPERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/normalize-helpers/")
endif(WIN32)
add_definitions(-DPVRUSH_NORMALIZE_HELPERS_DIR="${PVRUSH_NORMALIZE_HELPERS_DIR}")

# Set PVRUSH_DECODERS_DIR
if(WIN32)
        set(PVRUSH_DECODERS_DIR "decoders")
else(WIN32)
        set(PVRUSH_DECODERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/decoders/")
endif(WIN32)
add_definitions(-DPVRUSH_DECODERS_DIR="${PVRUSH_DECODERS_DIR}")

# Set PVRUSH_CACHE_DIR
if(WIN32)
        set(PVRUSH_CACHE_DIR "cache")
else(WIN32)
        set(PVRUSH_CACHE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/cache/")
endif(WIN32)
add_definitions(-DPVRUSH_CACHE_DIR="${PVRUSH_CACHE_DIR}")

# Set PVRUSH_INPUTTYPE_DIR
if(WIN32)
	set(PVRUSH_INPUTTYPE_DIR "input-types")
else(WIN32)
	set(PVRUSH_INPUTTYPE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/input-types/")
endif(WIN32)
add_definitions(-DPVRUSH_INPUTTYPE_DIR="${PVRUSH_INPUTTYPE_DIR}")

# Set PVRUSH_SOURCE_DIR
if(WIN32)
	set(PVRUSH_SOURCE_DIR "sources")
else(WIN32)
	set(PVRUSH_SOURCE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/sources/")
endif(WIN32)
add_definitions(-DPVRUSH_SOURCE_DIR="${PVRUSH_SOURCE_DIR}")
########################################
# END: SET VARIOUS PLUGINS DIRECTORIES #
########################################


add_subdirectory(src)
add_subdirectory(plugins)

set(PVKERNEL_PLUGINS_LIST ${PLUGINS_LIST})

#add_subdirectory(bindings)
#add_subdirectory(utils)

add_subdirectory(tests)
propagate_tests_targets_to_parent()
