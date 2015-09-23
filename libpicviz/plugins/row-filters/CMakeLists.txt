#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

###############################################################################
# Row filters
###############################################################################

###############################################################################
# Define row filter plugins to distribute
###############################################################################
set(FILTERS_DIST_LIST
	row_filter_axes_bind
)

set(FILTERS_DIST_FILES
)

# Define extra lib
set(LINK_LIBRARIES picviz pvkernel)

###############################################################################
# Axes bind row filter
###############################################################################
# Define ressource
SET(SRC_AXES
axes-bind/PVRFFAxesBind.cpp
axes-bind/register.cpp
)

# Create lib
add_library(row_filter_axes_bind SHARED ${SRC_AXES})
target_link_libraries(row_filter_axes_bind ${LINK_LIBRARIES})

###############################################################################
# Install ressources and plugins
###############################################################################

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION row-filters
                                     RUNTIME DESTINATION row-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION row-filters)
