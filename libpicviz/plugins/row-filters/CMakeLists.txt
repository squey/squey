#
# Row filters
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)


link_directories(${picviz-inspector_SOURCE_DIR}/libpicviz/src/ ${picviz-inspector_SOURCE_DIR}/libpvfilter/src/)

set(FILTERS_DIST_LIST
row_filter_axes_bind
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

add_library(row_filter_axes_bind SHARED axes-bind/PVRFFAxesBind.cpp axes-bind/register.cpp)
target_link_libraries(row_filter_axes_bind ${LINK_LIBRARIES})

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION row-filters
                                     RUNTIME DESTINATION row-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION row-filters)
