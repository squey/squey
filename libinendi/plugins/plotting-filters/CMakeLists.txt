#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

###############################################################################
# Plotting filters
###############################################################################

###############################################################################
# Define plotting filter plugns to distribute
###############################################################################

set(FILTERS_DIST_LIST
plotting_filter_noprocess
plotting_filter_all_minmax
plotting_filter_all_log_minmax
plotting_filter_integer_port
plotting_filter_time_default
)

set(FILTERS_DIST_FILES
)

# Extra libs
set(LINK_LIBRARIES inendi pvkernel)

###############################################################################
# No process filter
###############################################################################
# Define ressources
SET(SRC_NOPROCESS
noprocess/PVPlottingFilterNoprocess.cpp
noprocess/register.cpp
)

#Create lib
add_library(plotting_filter_noprocess SHARED ${SRC_NOPROCESS})
target_link_libraries(plotting_filter_noprocess ${LINK_LIBRARIES})

###############################################################################
# Min max filter
###############################################################################
# Define ressources
SET(SRC_MINMAX
all_minmax/PVPlottingFilterMinmax.cpp
all_minmax/register.cpp
)

#Create lib
add_library(plotting_filter_all_minmax SHARED ${SRC_MINMAX})
target_link_libraries(plotting_filter_all_minmax ${LINK_LIBRARIES})

###############################################################################
# Log min max filter
###############################################################################
# Define ressources
SET(SRC_LOGMINMAX
all_log_minmax/PVPlottingFilterLogMinmax.cpp
all_log_minmax/register.cpp
)

#Create lib
add_library(plotting_filter_all_log_minmax SHARED ${SRC_LOGMINMAX})
target_link_libraries(plotting_filter_all_log_minmax ${LINK_LIBRARIES})

###############################################################################
# Port filter
###############################################################################
# Define ressources
SET(SRC_PORT
integer_port/PVPlottingFilterIntegerPort.cpp
integer_port/register.cpp
)

#Create lib
add_library(plotting_filter_integer_port SHARED ${SRC_PORT})
target_link_libraries(plotting_filter_integer_port ${LINK_LIBRARIES})

###############################################################################
# Time filter
###############################################################################
# Define ressources
SET(SRC_TIME
time_default/PVPlottingFilterTimeDefault.cpp
time_default/register.cpp
)

#Create lib
add_library(plotting_filter_time_default SHARED ${SRC_TIME})
target_link_libraries(plotting_filter_time_default ${LINK_LIBRARIES})

###############################################################################
# Install plugins and ressources
###############################################################################
INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION plotting-filters
                                     RUNTIME DESTINATION plotting-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION plotting-filters)
