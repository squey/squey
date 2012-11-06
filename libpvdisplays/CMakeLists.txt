#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpvdisplays)

include(CTest)
set(CTEST_PROJECT_NAME "PVDisplays")

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})

# Set PICVIZ_DISPLAYS_DIR
if(WIN32)
        set(PICVIZ_DISPLAYS_DIR "displays")
else(WIN32)
        set(PICVIZ_DISPLAYS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/displays/")
endif(WIN32)
add_definitions(-DPICVIZ_DISPLAYS_DIR="${PICVIZ_DISPLAYS_DIR}")

add_subdirectory(src)
add_subdirectory(plugins)
add_subdirectory(tests)
