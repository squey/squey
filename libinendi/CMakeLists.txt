#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

project(libinendi)

################################################
# QT5 related include and check
################################################

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(SYSTEM ${Qt5Xml_INCLUDE_DIRS})

# For QOpenGLWidget
find_package(Qt5OpenGL REQUIRED)
include_directories(SYSTEM ${Qt5OpenGL_INCLUDE_DIRS})

# For QColor
find_package(Qt5Gui REQUIRED)
include_directories(SYSTEM ${Qt5Gui_INCLUDE_DIRS})

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(SYSTEM ${Qt5Widgets_INCLUDE_DIRS})

###################################################################
# Set plugins install path.
###################################################################
add_definitions(-DINENDI_PLUGIN_PATH="${INENDI_PLUGIN}")

###################################################################
# Add include directories for inendi's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})
include_directories(SYSTEM AFTER ${ICU_INCLUDE})
include_directories(SYSTEM AFTER ${SIGCPP_INCLUDE_DIRS})

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)

add_subdirectory(tests)
