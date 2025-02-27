#
# @file
#

project(gui-qt)

add_definitions(-DBUILD_SQUEY)

include(CTest)
set(CTEST_PROJECT_NAME "GuiQt")

################################################
# QT6 related include and check
################################################

# For QApplication
find_package(Qt6Widgets REQUIRED)
include_directories(SYSTEM ${Qt6Widgets_INCLUDE_DIRS})

# For QDomElement
find_package(Qt6Xml REQUIRED)
include_directories(SYSTEM ${Qt6Xml_INCLUDE_DIRS})

# For QDesktopWidget
find_package(Qt6Widgets REQUIRED)
include_directories(SYSTEM ${Qt6Widgets_INCLUDE_DIRS})

# For QDBus
find_package(Qt6DBus REQUIRED)
include_directories(SYSTEM ${Qt6DBus_INCLUDE_DIRS})

# For Qt6Core5Compat
find_package(Qt6Core5Compat REQUIRED)
include_directories(SYSTEM ${Qt6Core5Compat_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

################################################
# Defined included directory
################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${SQUEY_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS}/pvguiqt/format-builder)
include_directories(AFTER ${PVLICENSE_INCLUDE_DIR})
include_directories(AFTER SYSTEM ${SIGCPP_INCLUDE_DIRS})

include_directories(AFTER include)


add_subdirectory(src)
add_subdirectory(tests)
