#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpvguiqt)

include(CTest)
set(CTEST_PROJECT_NAME "PVGuiQT")

################################################
# QT5 related include and check
################################################

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(${Qt5Xml_INCLUDE_DIRS})

# For QAction
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

# For QConcurrent
find_package(Qt5Concurrent REQUIRED)
include_directories(${Qt5Concurrent_INCLUDE_DIRS})

# For QGLWidget
find_package(Qt5OpenGL REQUIRED)
include_directories(${Qt5OpenGL_INCLUDE_DIRS})

#################################################
# Add include directories for picviz's libs
#################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)
propagate_tests_targets_to_parent()
