#
# @file
#

project(libpvparallelview)

################################################
# QT5 related include and check
################################################

# For QColors
find_package(Qt5Gui REQUIRED)
include_directories(SYSTEM ${Qt5Gui_INCLUDE_DIRS})

# For QGraphicsItem
find_package(Qt5Widgets REQUIRED)
include_directories(SYSTEM ${Qt5Widgets_INCLUDE_DIRS})

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(SYSTEM ${Qt5Xml_INCLUDE_DIRS})

# For QGLWidget
find_package(Qt5OpenGL REQUIRED)
include_directories(SYSTEM ${Qt5OpenGL_INCLUDE_DIRS})

# For KItemModels
find_package(KF5ItemModels REQUIRED)

# For KSyntaxHighlighting
find_package(KF5SyntaxHighlighting REQUIRED)

#################################################
# Add include directories for inendi's libs
#################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(SYSTEM AFTER ${SIGCPP_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)
