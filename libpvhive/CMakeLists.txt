#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpvhive)

include(CTest)
set(CTEST_PROJECT_NAME "PVHive")

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)
