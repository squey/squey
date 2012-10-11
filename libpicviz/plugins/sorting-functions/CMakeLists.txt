#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#
# Layer filters
#

set(FUNCTIONS_DIST_LIST
sorting_float_default
sorting_integer_default
)

set(FUNCTIONS_DIST_FILES
)

set(LINK_LIBRARIES picviz pvkernel ${QT_LIBRARIES})

add_library(sorting_float_default SHARED float_default/PVFloatDefaultSortingFunc.cpp float_default/register.cpp)
target_link_libraries(sorting_float_default ${LINK_LIBRARIES})

add_library(sorting_integer_default SHARED integer_default/PVIntegerDefaultSortingFunc.cpp integer_default/register.cpp)
target_link_libraries(sorting_integer_default ${LINK_LIBRARIES})

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION sorting-functions
                                     RUNTIME DESTINATION sorting-functions)

INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION sorting-functions)
