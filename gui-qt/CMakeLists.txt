#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(gui-qt)

option(USE_OWN_LIBCURL "We have problems with quite some distributions with libcurl. Use installed version in /opt/curl/" TRUE)

if (WIN32)
    list( APPEND ALL_SOURCES src/resources/winicon.rc )
    source_group( "Resources" FILES Resources/icon.rc )
endif(WIN32)

add_definitions(${OpenMP_CXX_FLAGS})
add_definitions(${COMPILER_OPTIMIZE_FLAGS})

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})

include_directories(AFTER src/include)
include_directories(AFTER src/widgets/format-builder/include)

add_subdirectory(src)
