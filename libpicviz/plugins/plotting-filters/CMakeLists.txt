#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#
# Plotting filters
#

set(FILTERS_DIST_LIST
plotting_filter_noprocess
plotting_filter_all_minmax
plotting_filter_all_log_minmax
#plotting_filter_all_divide
plotting_filter_integer_port
plotting_filter_time_default
#plotting_filter_ipv4_default
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES picviz pvkernel)

add_library(plotting_filter_noprocess SHARED noprocess/PVPlottingFilterNoprocess.cpp noprocess/register.cpp)
target_link_libraries(plotting_filter_noprocess ${LINK_LIBRARIES})

add_library(plotting_filter_all_minmax SHARED all_minmax/PVPlottingFilterMinmax.cpp all_minmax/register.cpp)
target_link_libraries(plotting_filter_all_minmax ${LINK_LIBRARIES})

add_library(plotting_filter_all_log_minmax SHARED all_log_minmax/PVPlottingFilterLogMinmax.cpp all_log_minmax/register.cpp)
target_link_libraries(plotting_filter_all_log_minmax ${LINK_LIBRARIES})

#add_library(plotting_filter_float_igr SHARED float_igr/PVPlottingFilterFloatIGR.cpp float_igr/register.cpp)
#target_link_libraries(plotting_filter_float_igr ${LINK_LIBRARIES})

#add_library(plotting_filter_all_divide SHARED all_divide/PVPlottingFilterAllDivide.cpp all_divide/register.cpp)
#target_link_libraries(plotting_filter_all_divide ${LINK_LIBRARIES})

#add_library(plotting_filter_ipv4_default SHARED ipv4_default/PVPlottingFilterIpv4Default.cpp ipv4_default/register.cpp)
#target_link_libraries(plotting_filter_ipv4_default ${LINK_LIBRARIES})

add_library(plotting_filter_integer_port SHARED integer_port/PVPlottingFilterIntegerPort.cpp integer_port/register.cpp)
target_link_libraries(plotting_filter_integer_port ${LINK_LIBRARIES})

add_library(plotting_filter_time_default SHARED time_default/PVPlottingFilterTimeDefault.cpp time_default/register.cpp)
target_link_libraries(plotting_filter_time_default ${LINK_LIBRARIES})

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION plotting-filters
                                     RUNTIME DESTINATION plotting-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION plotting-filters)
