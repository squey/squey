#
# @file
#

project(libpvparallelview)

################################################
# QT6 related include and check
################################################

# For QColors
find_package(Qt6Gui REQUIRED)
include_directories(SYSTEM ${Qt6Gui_INCLUDE_DIRS})

# For QGraphicsItem
find_package(Qt6Widgets REQUIRED)
include_directories(SYSTEM ${Qt6Widgets_INCLUDE_DIRS})

# For QDomElement
find_package(Qt6Xml REQUIRED)
include_directories(SYSTEM ${Qt6Xml_INCLUDE_DIRS})

# For QGLWidget
find_package(Qt6OpenGL REQUIRED)
include_directories(SYSTEM ${Qt6OpenGL_INCLUDE_DIRS})
find_package(Qt6OpenGLWidgets REQUIRED)
include_directories(SYSTEM ${Qt6OpenGLWidgets_INCLUDE_DIRS})

# For KItemModels
find_package(KF6ItemModels REQUIRED)
include_directories(AFTER /app/include/KF6/KItemModels/)

# For KSyntaxHighlighting
find_package(KF5SyntaxHighlighting REQUIRED)

#################################################
# Add include directories for squey's libs
#################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${SQUEY_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(SYSTEM AFTER ${SIGCPP_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)
