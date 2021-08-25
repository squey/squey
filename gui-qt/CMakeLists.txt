#
# @file
#

project(gui-qt)

include(CTest)
set(CTEST_PROJECT_NAME "GuiQt")

################################################
# QT5 related include and check
################################################

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(SYSTEM ${Qt5Widgets_INCLUDE_DIRS})

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(SYSTEM ${Qt5Xml_INCLUDE_DIRS})

# For QDesktopWidget
find_package(Qt5Widgets REQUIRED)
include_directories(SYSTEM ${Qt5Widgets_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

################################################
# Defined included directory
################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})
include_directories(AFTER ${PVLICENSE_INCLUDE_DIR})
include_directories(SYSTEM AFTER ${SIGCPP_INCLUDE_DIRS})

include_directories(AFTER src/include)
include_directories(AFTER src/widgets/format-builder/include)

add_subdirectory(src)
