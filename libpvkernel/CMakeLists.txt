#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

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

##################################################################
# set installation path for query builder and plugins
##################################################################
install(DIRECTORY ${inendi-inspector_SOURCE_DIR}/libpvkernel/src/widgets/querybuilder DESTINATION ${SHARE_RESSOURCES})
add_definitions(-DINENDI_QUERY_BUILDER="${SHARE_RESSOURCES}/querybuilder")
add_definitions(-DPVKERNEL_PLUGIN_PATH="${PVKERNEL_PLUGIN}")

###################################################################
# Add include directories for inendi's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})
include_directories(AFTER ${ARCHIVE_INCLUDE_DIRS})
include_directories(AFTER ${BREAKPAD_INCLUDE_DIRS})
include_directories(AFTER ${Boost_INCLUDE_DIR})
include_directories(AFTER ${HWLOC_INCLUDE_DIRS})
include_directories(AFTER ${SIGCPP_INCLUDE_DIRS})
include_directories(AFTER "src/third_party/uchardetect")

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)
add_subdirectory(tools)

set(PVKERNEL_PLUGINS_LIST ${PLUGINS_LIST})

add_subdirectory(tests)
