#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(SAD_LIBRARY ${picviz-inspector_SOURCE_DIR}/helpers/libsad/src/libsad.so)

#set(CMAKE_VERBOSE_MAKEFILE ON)

# Set SAD_PLUGINS_DIR
if(WIN32)
	set(SAD_PLUGINS_DIR "plugins")
else(WIN32)
	set(SAD_PLUGINS_DIR "${CMAKE_INSTALL_PREFIX}/share/sad/plugins/")
endif(WIN32)
add_definitions(-DSAD_PLUGINS_DIR="${SAD_PLUGINS_DIR}")

#
# <Include directories>
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/helpers/libsad/src/include)
# include_directories(AFTER ${BOOST_INCLUDEDIR})
#
# </Include directories>
#

# Set SAD_PLUGINS_DIR
if(WIN32)
	set(SAD_PLUGINS_DIR "plugins")
else(WIN32)
	set(SAD_PLUGINS_DIR "${CMAKE_INSTALL_PREFIX}/share/sad/plugins/")
endif(WIN32)
add_definitions(-DSAD_PLUGINS_DIR="${SAD_PLUGINS_DIR}")

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})

add_subdirectory(src)
#add_subdirectory(plugins)
