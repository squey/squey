#
# @file
#

project(libinendi)

################################################
# QT6 related include and check
################################################

# For QDomElement
find_package(Qt6Xml REQUIRED)
include_directories(SYSTEM ${Qt6Xml_INCLUDE_DIRS})

# For QOpenGLWidget
find_package(Qt6OpenGL REQUIRED)
include_directories(SYSTEM ${Qt6OpenGL_INCLUDE_DIRS})

# For QColor
find_package(Qt6Gui REQUIRED)
include_directories(SYSTEM ${Qt6Gui_INCLUDE_DIRS})

# For QApplication
find_package(Qt6Widgets REQUIRED)
include_directories(SYSTEM ${Qt6Widgets_INCLUDE_DIRS})

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
