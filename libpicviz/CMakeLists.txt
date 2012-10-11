#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

project(libpicviz)

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})
include_directories(AFTER ${TULIP_INCLUDE_DIR})

##########################################
# BEGIN: SET VARIOUS PLUGINS DIRECTORIES #
##########################################
# Set PICVIZ_FUNCTIONS_DIR
if(WIN32)
        set(PICVIZ_FUNCTIONS_DIR "functions")
else(WIN32)
        set(PICVIZ_FUNCTIONS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/functions/")
endif(WIN32)
add_definitions(-DPICVIZ_FUNCTIONS_DIR="${PICVIZ_FUNCTIONS_DIR}")

# Set PICVIZ_FILTERS_DIR
if(WIN32)
        set(PICVIZ_FILTERS_DIR "filters")
else(WIN32)
        set(PICVIZ_FILTERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/filters/")
endif(WIN32)
add_definitions(-DPICVIZ_FILTERS_DIR="${PICVIZ_FILTERS_DIR}")

# Set PICVIZ_LAYER_FILTERS_DIR
if(WIN32)
	set(PICVIZ_LAYER_FILTERS_DIR "layer-filters")
else(WIN32)
	set(PICVIZ_LAYER_FILTERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/layer-filters/")
endif(WIN32)
add_definitions(-DPICVIZ_LAYER_FILTERS_DIR="${PICVIZ_LAYER_FILTERS_DIR}")

# Set PICVIZ_LAYER_FILTERS_DIR
if(WIN32)
	set(PICVIZ_LAYER_FILTERS_CONFIG_DIR "layer-filters\\conf")
else(WIN32)
	set(PICVIZ_LAYER_FILTERS_CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/layer-filters/conf/")
endif(WIN32)
add_definitions(-DPICVIZ_LAYER_FILTERS_CONFIG_DIR="${PICVIZ_LAYER_FILTERS_CONFIG_DIR}")

# Set PICVIZ_MAPPING_FILTERS_DIR
if(WIN32)
	set(PICVIZ_MAPPING_FILTERS_DIR "mapping-filters")
else(WIN32)
	set(PICVIZ_MAPPING_FILTERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/mapping-filters/")
endif(WIN32)
add_definitions(-DPICVIZ_MAPPING_FILTERS_DIR="${PICVIZ_MAPPING_FILTERS_DIR}")

# Set PICVIZ_PLOTTING_FILTERS_DIR
if(WIN32)
	set(PICVIZ_PLOTTING_FILTERS_DIR "plotting-filters")
else(WIN32)
	set(PICVIZ_PLOTTING_FILTERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/plotting-filters/")
endif(WIN32)
add_definitions(-DPICVIZ_PLOTTING_FILTERS_DIR="${PICVIZ_PLOTTING_FILTERS_DIR}")

# Set PICVIZ_ROW_FILTERS_DIR
if(WIN32)
	set(PICVIZ_ROW_FILTERS_DIR "row-filters")
else(WIN32)
	set(PICVIZ_ROW_FILTERS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/row-filters/")
endif(WIN32)
add_definitions(-DPICVIZ_ROW_FILTERS_DIR="${PICVIZ_ROW_FILTERS_DIR}")

# Set PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR
if(WIN32)
	set(PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR "axis-computation")
else(WIN32)
	set(PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/axis-computation/")
endif(WIN32)
add_definitions(-DPICVIZ_AXIS_COMPUTATION_PLUGINS_DIR="${PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR}")

# Set PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR
if(WIN32)
	set(PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR "sorting-functions")
else(WIN32)
	set(PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/sorting-functions")
endif(WIN32)
add_definitions(-DPICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR="${PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR}")

add_definitions(${OpenMP_CXX_FLAGS})
add_definitions(${COMPILER_OPTIMIZE_FLAGS})


# #
# # PCRE
# if(WIN32)
#     set(PCRE_FILES_DLL "C:\\\\dev\\\\GnuWin32\\\\bin\\\\pcre3.dll" "C:\\\\dev\\\\GnuWin32\\\\bin\\\\pcreposix3.dll") # For CPack; NSIS needs double escape
# 	set(PCRE_INCLUDE_DIR "C:\\dev\\GnuWin32\\include")
# 	set(PCRE_PCREPOSIX_LIBRARY "C:\\dev\\GnuWin32\\lib\\pcreposix.lib")
# 	set(PCRE_PCRE_LIBRARY "C:\\dev\\GnuWin32\\lib\\pcre.lib")
# else(WIN32)
# 	FIND_PACKAGE(PCRE REQUIRED)
# endif(WIN32)
# #
# # WinLicense
# # 
# if(WIN32)
# 	set(WINLICENSE_LIBRARY "C:\\dev\\WinlicenseSDK.lib")
# endif(WIN32)

add_subdirectory(src)
add_subdirectory(plugins)

add_subdirectory(tests)
