#
# Layer filters
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

# PCAP
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

link_directories(${picviz-inspector_SOURCE_DIR}/libpicviz/src/ ${picviz-inspector_SOURCE_DIR}/libpvfilter/src/)

set(FILTERS_DIST_LIST
layer_filter_axis_gradient
layer_filter_heatline
layer_filter_find_duplicates
layer_filter_multiple_search
layer_filter_search
layer_filter_selection_search
layer_filter_create_layers
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

add_library(layer_filter_multiple_search SHARED multiple-search/PVLayerFilterMultipleSearch.cpp multiple-search/register.cpp)
target_link_libraries(layer_filter_multiple_search ${LINK_LIBRARIES})

add_library(layer_filter_selection_search SHARED selection-search/PVLayerFilterSelectionSearch.cpp selection-search/register.cpp)
target_link_libraries(layer_filter_selection_search ${LINK_LIBRARIES})

add_library(layer_filter_search SHARED search/PVLayerFilterSearch.cpp search/register.cpp)
target_link_libraries(layer_filter_search ${LINK_LIBRARIES})

add_library(layer_filter_find_duplicates SHARED find-duplicates/PVLayerFilterFindDuplicates.cpp find-duplicates/register.cpp)
target_link_libraries(layer_filter_find_duplicates ${LINK_LIBRARIES})

add_library(layer_filter_create_layers SHARED create-layers/PVLayerFilterCreateLayers.cpp create-layers/parse-config.cpp create-layers/register.cpp)
target_link_libraries(layer_filter_create_layers ${LINK_LIBRARIES})

add_library(layer_filter_find_same_values SHARED find-same-values/PVLayerFilterFindSameValues.cpp find-same-values/register.cpp)
target_link_libraries(layer_filter_find_same_values ${LINK_LIBRARIES})

if(PYTHONLIBS_FOUND)
	# AG: BEFORE here is important, or boost does not include the good patchlevel.h (from Python), and does not
	# find the good Python version !!
	include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
	add_library(layer_filter_snort_for_logs SHARED snort-for-logs/PVLayerFilterSnortForLogs.cpp snort-for-logs/register.cpp)
	target_link_libraries(layer_filter_snort_for_logs ${LINK_LIBRARIES} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES_PYTHON})
	set(FILTERS_DIST_LIST
	${FILTERS_DIST_LIST}
	layer_filter_snort_for_logs
	)
endif(PYTHONLIBS_FOUND)

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION layer-filters
                                     RUNTIME DESTINATION layer-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION layer-filters)
