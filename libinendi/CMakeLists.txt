#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

project(libinendi)

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
# Add include directories for inendi's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})

##########################################
# BEGIN: SET VARIOUS PLUGINS DIRECTORIES #
##########################################
# Set INENDI_LAYER_FILTERS_DIR
if (CMAKE_BUILD_TYPE_LOWER STREQUAL "distrib")
	set(INENDI_LAYER_FILTERS_DIR "layer-filters")
else()
	set(INENDI_LAYER_FILTERS_DIR "../../libinendi/plugins/layer-filters/")
endif()
add_definitions(-DINENDI_LAYER_FILTERS_DIR="${INENDI_LAYER_FILTERS_DIR}")

# Set INENDI_MAPPING_FILTERS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "distrib")
	set(INENDI_MAPPING_FILTERS_DIR "mapping-filters")
else()
	set(INENDI_MAPPING_FILTERS_DIR "../../libinendi/plugins/mapping-filters/")
endif()
add_definitions(-DINENDI_MAPPING_FILTERS_DIR="${INENDI_MAPPING_FILTERS_DIR}")

# Set INENDI_PLOTTING_FILTERS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "distrib")
	set(INENDI_PLOTTING_FILTERS_DIR "plotting-filters")
else()
	set(INENDI_PLOTTING_FILTERS_DIR "../../libinendi/plugins/plotting-filters/")
endif()
add_definitions(-DINENDI_PLOTTING_FILTERS_DIR="${INENDI_PLOTTING_FILTERS_DIR}")

# Set INENDI_AXIS_COMPUTATION_PLUGINS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "distrib")
	set(INENDI_AXIS_COMPUTATION_PLUGINS_DIR "axis-computation")
else()
	set(INENDI_AXIS_COMPUTATION_PLUGINS_DIR "../../libinendi/plugins/axis-computation/")
endif()
add_definitions(-DINENDI_AXIS_COMPUTATION_PLUGINS_DIR="${INENDI_AXIS_COMPUTATION_PLUGINS_DIR}")

# Set INENDI_SORTING_FUNCTIONS_PLUGINS_DIR
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "distrib")
	set(INENDI_SORTING_FUNCTIONS_PLUGINS_DIR "sorting-functions")
else()
	set(INENDI_SORTING_FUNCTIONS_PLUGINS_DIR "../../libinendi/plugins/sorting-functions")
endif()
add_definitions(-DINENDI_SORTING_FUNCTIONS_PLUGINS_DIR="${INENDI_SORTING_FUNCTIONS_PLUGINS_DIR}")

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)

add_subdirectory(tests)
