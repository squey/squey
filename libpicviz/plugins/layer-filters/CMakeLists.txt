#
# Layer filters
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

# PCAP
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

link_directories(${picviz-inspector_SOURCE_DIR}/libpicviz/src/ ${picviz-inspector_SOURCE_DIR}/libpvfilter/src/)

set(FILTERS_DIST_LIST
layer_filter_axis_gradient
layer_filter_diff
layer_filter_heatline
layer_filter_find_duplicates
layer_filter_find_not_duplicates
layer_filter_search
layer_filter_webmail
layer_filter_attacks
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

add_library(layer_filter_axis_gradient SHARED axis-gradient/PVLayerFilterAxisGradient.cpp axis-gradient/register.cpp)
target_link_libraries(layer_filter_axis_gradient ${LINK_LIBRARIES})

add_library(layer_filter_diff SHARED diff/PVLayerFilterDiff.cpp diff/register.cpp)
target_link_libraries(layer_filter_diff ${LINK_LIBRARIES})

add_library(layer_filter_heatline SHARED heatline/PVLayerFilterHeatline.cpp heatline/register.cpp)
target_link_libraries(layer_filter_heatline ${LINK_LIBRARIES})

add_library(layer_filter_search SHARED search/PVLayerFilterSearch.cpp search/register.cpp)
target_link_libraries(layer_filter_search ${LINK_LIBRARIES})

add_library(layer_filter_find_duplicates SHARED find-duplicates/PVLayerFilterFindDuplicates.cpp find-duplicates/register.cpp)
target_link_libraries(layer_filter_find_duplicates ${LINK_LIBRARIES})

add_library(layer_filter_find_not_duplicates SHARED find-not-duplicates/PVLayerFilterFindNotDuplicates.cpp find-not-duplicates/register.cpp)
target_link_libraries(layer_filter_find_not_duplicates ${LINK_LIBRARIES})

add_library(layer_filter_webmail SHARED webmail-finder/PVLayerFilterWebmailFinder.cpp webmail-finder/register.cpp)
target_link_libraries(layer_filter_webmail ${LINK_LIBRARIES})

add_library(layer_filter_attacks SHARED find-attacks/PVLayerFilterFindAttacks.cpp find-attacks/register.cpp)
target_link_libraries(layer_filter_attacks ${LINK_LIBRARIES})

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION layer-filters
                                     RUNTIME DESTINATION layer-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION layer-filters)
