#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpvguiqt)

include(CTest)
set(CTEST_PROJECT_NAME "PVGuiQT")

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})
include_directories(AFTER ${QT_QTOPENGL_INCLUDE_DIR})
include_directories(AFTER ${QT_UI_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(tests)
