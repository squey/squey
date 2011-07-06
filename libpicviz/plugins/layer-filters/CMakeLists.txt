#
# Layer filters
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)

include_directories(AFTER ${APR_INCLUDES})
# PCRE
include_directories(AFTER ${PCRE_INCLUDE_DIR})
# PCAP
include_directories(AFTER ${PCAP_INCLUDE_DIRS})
# DUMBNET
include_directories(AFTER ${DUMBNET_INCLUDE_DIRS})

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/)

set(FILTERS_DIST_LIST
layer_filter_axis_gradient
layer_filter_diff
layer_filter_heatline
layer_filter_search
layer_filter_webmail
)

set(FILTERS_DIST_FILES
)

add_library(layer_filter_axis_gradient SHARED axis-gradient/PVLayerFilterAxisGradient.cpp axis-gradient/register.cpp)
target_link_libraries(layer_filter_axis_gradient picviz pvfilter)

add_library(layer_filter_diff SHARED diff/PVLayerFilterDiff.cpp diff/register.cpp)
target_link_libraries(layer_filter_diff picviz pvfilter)

add_library(layer_filter_heatline SHARED heatline/PVLayerFilterHeatline.cpp heatline/register.cpp)
target_link_libraries(layer_filter_heatline picviz pvfilter)

add_library(layer_filter_search SHARED search/PVLayerFilterSearch.cpp search/register.cpp)
target_link_libraries(layer_filter_search picviz pvfilter)

add_library(layer_filter_webmail SHARED webmail-finder/PVLayerFilterWebmailFinder.cpp webmail-finder/register.cpp)
target_link_libraries(layer_filter_webmail picviz pvfilter)

INSTALL(TARGETS ${FILTERS_DIST_LIST} DESTINATION layer-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION layer-filters)
