#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

###############################################################################
# Layer filters
###############################################################################

###############################################################################
# Define plugins to distribute
###############################################################################

set(FILTERS_DIST_LIST
layer_filter_axis_gradient
layer_filter_heatline
layer_filter_multiple_search
)

set(FILTERS_DIST_FILES
create-layers/create-layers.conf
)

set(LINK_LIBRARIES inendi pvkernel)

###############################################################################
# Axis Gradient layer filter
###############################################################################
# Define ressources
SET(GRADIENT_SRC
axis-gradient/PVLayerFilterAxisGradient.cpp
axis-gradient/register.cpp
)

# Create lib
add_library(layer_filter_axis_gradient SHARED ${GRADIENT_SRC})
target_link_libraries(layer_filter_axis_gradient ${LINK_LIBRARIES})

###############################################################################
# Heatline layer filter
###############################################################################
# Define ressources
SET(HEATLINE_SRC
heatline/PVLayerFilterHeatline.cpp
heatline/register.cpp
)

# Create lib
add_library(layer_filter_heatline SHARED ${HEATLINE_SRC})
target_link_libraries(layer_filter_heatline ${LINK_LIBRARIES})

###############################################################################
# Multiple Search layer filter
###############################################################################
# Define ressources
SET(MULTISEARCH_SRC
multiple-search/PVLayerFilterMultipleSearch.cpp
multiple-search/register.cpp
)

# Create lib
add_library(layer_filter_multiple_search SHARED ${MULTISEARCH_SRC})
target_link_libraries(layer_filter_multiple_search ${LINK_LIBRARIES} ${PCRE_LIBRARIES})

###############################################################################
# Install plugins and ressources
###############################################################################

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION layer-filters
                                     RUNTIME DESTINATION layer-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION .)
