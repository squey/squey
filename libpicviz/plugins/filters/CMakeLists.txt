#
# Filters
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

link_directories(${picviz-inspector_SOURCE_DIR}/libpicviz/src/)

set(FILTERS_DIST_LIST
#filter_diff
#filter_heatline_color
#filter_heatline_select
# filter_search
#filter_snort
# filter_log_heatline_color
# filter_log_heatline_select
#filter_select_trojan_ports
#filter_dshield
)

set(FILTERS_DIST_FILES
dshield-ipascii.txt
)

# add_library(filter_diff SHARED diff.c)
# target_link_libraries(filter_diff ${APR_LIBS} picviz)

# add_library(filter_heatline_color SHARED heatline/core.c heatline/colorize.c)
# target_link_libraries(filter_heatline_color ${APR_LIBS} picviz)

# add_library(filter_heatline_select SHARED heatline/core.c heatline/select.c)
# target_link_libraries(filter_heatline_select ${APR_LIBS} picviz)

add_library(filter_search SHARED search/search.cpp)
target_link_libraries(filter_search ${PICVIZ_LIBRARIES_WITHOUT_PICVIZ} ${APR_LIBS} picviz)

# add_library(filter_snort SHARED snort/snort.c)
# target_link_libraries(filter_snort ${APR_LIBS} picviz)

# add_library(filter_above-value SHARED above-value/above-value.c)
# target_link_libraries(filter_above-value ${APR_LIBS} picviz)

add_library(filter_log_heatline_color SHARED log-heatline/core.cpp log-heatline/colorize.cpp)
target_link_libraries(filter_log_heatline_color ${APR_LIBS} picviz)

add_library(filter_log_heatline_select SHARED log-heatline/core.cpp log-heatline/select.cpp)
target_link_libraries(filter_log_heatline_select ${APR_LIBS} picviz)

# add_library(filter_select_trojan_ports SHARED trojan-ports/select-trojan-ports.c)
# target_link_libraries(filter_select_trojan_ports ${APR_LIBS} picviz)

# add_library(filter_dshield SHARED dshield/dshield.c)
# target_link_libraries(filter_dshield ${APR_LIBS} picviz)

#add_library(filter_blacklist_toulouse SHARED blacklist-toulouse/blacklist-toulouse.c)
#target_link_libraries(filter_blacklist_toulouse ${APR_LIBS} picviz)

add_library(filter_axis_gradient SHARED axis-gradient/axis-gradient.cpp)
target_link_libraries(filter_axis_gradient picviz)

INSTALL(TARGETS ${FILTERS_DIST_LIST} DESTINATION filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION filters)
