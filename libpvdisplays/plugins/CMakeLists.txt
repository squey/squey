#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#
# Mapping filters
#
include_directories(AFTER ${ICU_INCLUDE})

set(FILTERS_DIST_LIST
)

set(FILTERS_DIST_FILES
)

set(COMMON_LINK_LIBRARIES pvdisplays pvhive picviz pvkernel ${QT_LIBRARIES})

include_directories(${PVHIVE_INCLUDE_DIRS})

#add_library(display_listing SHARED listing/PVDisplayViewListing.cpp listing/register.cpp)
#target_link_libraries(display_listing pvguiqt ${LINK_LIBRARIES})


INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION mapping-filters
                                     RUNTIME DESTINATION mapping-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION mapping-filters)
