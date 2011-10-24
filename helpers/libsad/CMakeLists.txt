set(SAD_LIBRARY ${Picviz_Inspector_SOURCE_DIR}/helpers/libsad/src/libsad.so)

set(CMAKE_VERBOSE_MAKEFILE ON)

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
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/helpers/libsad/src/include)
# include(${QT_USE_FILE})
# include_directories(AFTER ${BOOST_INCLUDEDIR})
#
# </Include directories>
#

# message(STATUS "QtCore include dirs: " ${QT_QTCORE_INCLUDE_DIR})

# Set SAD_PLUGINS_DIR
if(WIN32)
	set(SAD_PLUGINS_DIR "plugins")
else(WIN32)
	set(SAD_PLUGINS_DIR "${CMAKE_INSTALL_PREFIX}/share/sad/plugins/")
endif(WIN32)
add_definitions(-DSAD_PLUGINS_DIR="${SAD_PLUGINS_DIR}")


add_subdirectory(src)
#add_subdirectory(plugins)
