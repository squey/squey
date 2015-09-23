#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpicviz)

################################################
# QT5 related include and check
################################################

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(${Qt5Xml_INCLUDE_DIRS})

# For QColor
find_package(Qt5Gui REQUIRED)
include_directories(${Qt5Gui_INCLUDE_DIRS})

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

###################################################################
# Add include directories for picviz's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})
if(ENABLE_CORRELATION)
        include_directories(AFTER ${TULIP_INCLUDE_DIR})
endif()

##########################################
# BEGIN: SET VARIOUS PLUGINS DIRECTORIES #
##########################################
# Set PICVIZ_LAYER_FILTERS_DIR
if (CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_LAYER_FILTERS_DIR "layer-filters")
else()
	set(PICVIZ_LAYER_FILTERS_DIR "../../libpicviz/plugins/layer-filters/")
endif()
add_definitions(-DPICVIZ_LAYER_FILTERS_DIR="${PICVIZ_LAYER_FILTERS_DIR}")

# Set PICVIZ_MAPPING_FILTERS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_MAPPING_FILTERS_DIR "mapping-filters")
else()
	set(PICVIZ_MAPPING_FILTERS_DIR "../../libpicviz/plugins/mapping-filters/")
endif()
add_definitions(-DPICVIZ_MAPPING_FILTERS_DIR="${PICVIZ_MAPPING_FILTERS_DIR}")

# Set PICVIZ_PLOTTING_FILTERS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_PLOTTING_FILTERS_DIR "plotting-filters")
else()
	set(PICVIZ_PLOTTING_FILTERS_DIR "../../libpicviz/plugins/plotting-filters/")
endif()
add_definitions(-DPICVIZ_PLOTTING_FILTERS_DIR="${PICVIZ_PLOTTING_FILTERS_DIR}")

# Set PICVIZ_ROW_FILTERS_DIR
# This is unused for now as it require correlation enable
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_ROW_FILTERS_DIR "row-filters")
else()
	set(PICVIZ_ROW_FILTERS_DIR "../../libpicviz/plugins/row-filters/")
endif()
add_definitions(-DPICVIZ_ROW_FILTERS_DIR="${PICVIZ_ROW_FILTERS_DIR}")

# Set PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR "axis-computation")
else()
	set(PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR "../../libpicviz/plugins/axis-computation/")
endif()
add_definitions(-DPICVIZ_AXIS_COMPUTATION_PLUGINS_DIR="${PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR}")

# Set PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "relprotect")
	set(PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR "sorting-functions")
else()
	set(PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR "../../libpicviz/plugins/sorting-functions")
endif()
add_definitions(-DPICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR="${PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR}")

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)

add_subdirectory(tests)
propagate_tests_targets_to_parent()
