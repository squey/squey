#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

add_subdirectory(core)
add_subdirectory(rush)
add_subdirectory(widgets)
