#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(gui-qt)

include(CTest)
set(CTEST_PROJECT_NAME "GuiQt")

################################################
# QT5 related include and check
################################################

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

# For QNetworkAccessManager
find_package(Qt5Network REQUIRED)
include_directories(${Qt5Network_INCLUDE_DIRS})

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(${Qt5Xml_INCLUDE_DIRS})

# For QDesktopWidget
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(USE_OWN_LIBCURL "We have problems with quite some distributions with libcurl. Use installed version in /opt/curl/" TRUE)

if (WIN32)
    list( APPEND ALL_SOURCES src/resources/winicon.rc )
    source_group( "Resources" FILES Resources/icon.rc )
endif(WIN32)

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})

include_directories(AFTER src/include)
include_directories(AFTER src/widgets/format-builder/include)

add_subdirectory(src)
add_subdirectory(tests)
