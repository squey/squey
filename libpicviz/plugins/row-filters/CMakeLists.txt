#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#
# Row filters
#
set(FILTERS_DIST_LIST
	row_filter_axes_bind
	#row_filter_axes_bind_nearest_neighbors
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES picviz pvkernel ${QT_LIBRARIES})

add_library(row_filter_axes_bind SHARED axes-bind/PVRFFAxesBind.cpp axes-bind/register.cpp)
target_link_libraries(row_filter_axes_bind ${LINK_LIBRARIES})

#add_library(row_filter_axes_bind_nearest_neighbors SHARED axes-bind-nearest-neighbors/PVRFFAxesBindNearestNeighbors.cpp axes-bind-nearest-neighbors/register.cpp)
#target_link_libraries(row_filter_axes_bind_nearest_neighbors ${LINK_LIBRARIES})

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION row-filters
                                     RUNTIME DESTINATION row-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION row-filters)
