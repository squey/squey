#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include(CTest)
set(CTEST_PROJECT_NAME "PVGuiQT")

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})

add_definitions(${COMPILER_OPTIMIZE_FLAGS})
add_definitions(${OpenMP_CXX_FLAGS})

add_subdirectory(src)
add_subdirectory(tests)
