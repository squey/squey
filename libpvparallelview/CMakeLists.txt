#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpvparallelview)

set(QT_USE_QTOPENGL 1)

add_definitions(${COMPILER_OPTIMIZE_FLAGS})

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})

add_subdirectory(src)
#add_subdirectory(tests)
