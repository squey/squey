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
plotting_filter_default
plotting_filter_minmax
plotting_filter_log_minmax
plotting_filter_port
plotting_filter_enum
)

# Extra libs
set(LINK_LIBRARIES inendi pvkernel)

###############################################################################
# Default filter
###############################################################################
# Define ressources
SET(SRC_DEFAULT
default/PVPlottingFilterDefault.cpp
default/register.cpp
)

#Create lib
add_library(plotting_filter_default SHARED ${SRC_DEFAULT})
target_link_libraries(plotting_filter_default ${LINK_LIBRARIES})

###############################################################################
# Min max filter
###############################################################################
# Define ressources
SET(SRC_MINMAX
minmax/PVPlottingFilterMinmax.cpp
minmax/register.cpp
)

#Create lib
add_library(plotting_filter_minmax SHARED ${SRC_MINMAX})
target_link_libraries(plotting_filter_minmax ${LINK_LIBRARIES})

###############################################################################
# Log min max filter
###############################################################################
# Define ressources
SET(SRC_LOGMINMAX
log_minmax/PVPlottingFilterLogMinmax.cpp
log_minmax/register.cpp
)

#Create lib
add_library(plotting_filter_log_minmax SHARED ${SRC_LOGMINMAX})
target_link_libraries(plotting_filter_log_minmax ${LINK_LIBRARIES})

###############################################################################
# Port filter
###############################################################################
# Define ressources
SET(SRC_PORT
port/PVPlottingFilterPort.cpp
port/register.cpp
)

#Create lib
add_library(plotting_filter_port SHARED ${SRC_PORT})
target_link_libraries(plotting_filter_port ${LINK_LIBRARIES})

###############################################################################
# Enum filter
###############################################################################
# Define ressources
SET(SRC_ENUM
enum/PVPlottingFilterEnum.cpp
enum/register.cpp
)

#Create lib
add_library(plotting_filter_enum SHARED ${SRC_ENUM})
target_link_libraries(plotting_filter_enum ${LINK_LIBRARIES})

###############################################################################
# Install plugins and ressources
###############################################################################
INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION ${INENDI_PLUGIN}/plotting-filters
                                     RUNTIME DESTINATION ${INENDI_PLUGIN}/plotting-filters)
