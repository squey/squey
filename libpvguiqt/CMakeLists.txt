#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

project(libpvguiqt)

include(CTest)
set(CTEST_PROJECT_NAME "PVGuiQT")

set(COPYING_DIR_NAME "COPYING")
add_definitions(-DINENDI_COPYING_DIR="${SHARE_RESSOURCES}/${COPYING_DIR_NAME}")
install(DIRECTORY ${inendi-inspector_SOURCE_DIR}/${COPYING_DIR_NAME} DESTINATION ${SHARE_RESSOURCES})

################################################
# QT5 related include and check
################################################

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(SYSTEM ${Qt5Xml_INCLUDE_DIRS})

# For QAction
find_package(Qt5Widgets REQUIRED)
include_directories(SYSTEM ${Qt5Widgets_INCLUDE_DIRS})

# For QGLWidget
find_package(Qt5OpenGL REQUIRED)
include_directories(SYSTEM ${Qt5OpenGL_INCLUDE_DIRS})

#################################################
# Add include directories for inendi's libs
#################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})
include_directories(AFTER ${PVLICENSE_INCLUDE_DIR})
include_directories(SYSTEM AFTER ${SIGCPP_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)
