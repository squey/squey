#
# @file
#

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
layer_filter_errors_search
)

set(LINK_LIBRARIES pvsquey pvkernel)

###############################################################################
# Axis Gradient layer filter
###############################################################################
# Define ressources
set(GRADIENT_SRC
axis-gradient/PVLayerFilterAxisGradient.cpp
axis-gradient/register.cpp
)

# Create lib
add_library(layer_filter_axis_gradient SHARED ${GRADIENT_SRC})
target_link_libraries(layer_filter_axis_gradient PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Heatline layer filter
###############################################################################
# Define ressources
set(HEATLINE_SRC
heatline/PVLayerFilterHeatline.cpp
heatline/register.cpp
)

# Create lib
add_library(layer_filter_heatline SHARED ${HEATLINE_SRC})
target_link_libraries(layer_filter_heatline PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Multiple Search layer filter
###############################################################################
# Define ressources
set(MULTISEARCH_SRC
multiple-search/PVLayerFilterMultipleSearch.cpp
multiple-search/register.cpp
)

# Create lib
add_library(layer_filter_multiple_search SHARED ${MULTISEARCH_SRC})
target_link_libraries(layer_filter_multiple_search PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Errors Search layer filter
###############################################################################
# Define ressources
set(ERRORS_SRC
errors-search/PVLayerFilterErrorsSearch.cpp
errors-search/register.cpp
)

# Create lib
add_library(layer_filter_errors_search SHARED ${ERRORS_SRC})
target_link_libraries(layer_filter_errors_search PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Install plugins and ressources
###############################################################################

install(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION ${SQUEY_PLUGIN}/layer-filters
                                     RUNTIME DESTINATION ${SQUEY_PLUGIN}/layer-filters)
