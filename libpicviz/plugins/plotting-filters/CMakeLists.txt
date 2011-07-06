#
# Plotting filters
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/)

set(FILTERS_DIST_LIST
plotting_filter_noprocess
plotting_filter_all_minmax
plotting_filter_all_divide
plotting_filter_integer_port
plotting_filter_time_default
)

set(FILTERS_DIST_FILES
)

# 
# OpenMP
#

if (WIN32)
	add_definitions(/openmp)
else (WIN32)
	add_definitions(-fopenmp)
endif (WIN32)


add_library(plotting_filter_noprocess SHARED noprocess/PVPlottingFilterNoprocess.cpp noprocess/register.cpp)
target_link_libraries(plotting_filter_noprocess picviz pvfilter)

add_library(plotting_filter_all_minmax SHARED all_minmax/PVPlottingFilterMinmax.cpp all_minmax/register.cpp)
target_link_libraries(plotting_filter_all_minmax picviz pvfilter)

add_library(plotting_filter_all_divide SHARED all_divide/PVPlottingFilterAllDivide.cpp all_divide/register.cpp)
target_link_libraries(plotting_filter_all_divide picviz pvfilter)

add_library(plotting_filter_integer_port SHARED integer_port/PVPlottingFilterIntegerPort.cpp integer_port/register.cpp)
target_link_libraries(plotting_filter_integer_port picviz pvfilter)

add_library(plotting_filter_time_default SHARED time_default/PVPlottingFilterTimeDefault.cpp time_default/register.cpp)
target_link_libraries(plotting_filter_time_default picviz pvfilter)

INSTALL(TARGETS ${FILTERS_DIST_LIST} DESTINATION plotting-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION plotting-filters)
