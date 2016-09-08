#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

project(libpvdisplays)

################################################
# QT5 related include and check
################################################

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(SYSTEM ${Qt5Xml_INCLUDE_DIRS})

# For QAction
find_package(Qt5Widgets REQUIRED)
include_directories(SYSTEM ${Qt5Widgets_INCLUDE_DIRS})

#################################################
# Add include directories for inendi's libs
#################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})

add_subdirectory(src)
