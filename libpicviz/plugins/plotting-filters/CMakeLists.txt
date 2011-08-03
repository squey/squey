#
# Plotting filters
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)

set(FILTERS_DIST_LIST
plotting_filter_noprocess
plotting_filter_all_minmax
plotting_filter_all_divide
plotting_filter_integer_port
plotting_filter_time_default
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

add_library(plotting_filter_noprocess SHARED noprocess/PVPlottingFilterNoprocess.cpp noprocess/register.cpp)
target_link_libraries(plotting_filter_noprocess ${LINK_LIBRARIES})

add_library(plotting_filter_all_minmax SHARED all_minmax/PVPlottingFilterMinmax.cpp all_minmax/register.cpp)
target_link_libraries(plotting_filter_all_minmax ${LINK_LIBRARIES})

add_library(plotting_filter_all_divide SHARED all_divide/PVPlottingFilterAllDivide.cpp all_divide/register.cpp)
target_link_libraries(plotting_filter_all_divide ${LINK_LIBRARIES})

add_library(plotting_filter_integer_port SHARED integer_port/PVPlottingFilterIntegerPort.cpp integer_port/register.cpp)
target_link_libraries(plotting_filter_integer_port ${LINK_LIBRARIES})

add_library(plotting_filter_time_default SHARED time_default/PVPlottingFilterTimeDefault.cpp time_default/register.cpp)
target_link_libraries(plotting_filter_time_default ${LINK_LIBRARIES})

INSTALL(TARGETS ${FILTERS_DIST_LIST} DESTINATION plotting-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION plotting-filters)
