#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

###############################################################################
# Sorting functions
###############################################################################

###############################################################################
# Define plugins to distribute
###############################################################################

set(FUNCTIONS_DIST_LIST
sorting_float_default
sorting_integer_default
sorting_ipv4_default
)

set(FUNCTIONS_DIST_FILES
)

# Extra libs
set(LINK_LIBRARIES inendi pvkernel)

###############################################################################
# Float Sort
###############################################################################
# Define ressources
SET(SRC_FLOAT
float_default/PVFloatDefaultSortingFunc.cpp
float_default/register.cpp
)

#Create lib
add_library(sorting_float_default SHARED ${SRC_FLOAT})
target_link_libraries(sorting_float_default ${LINK_LIBRARIES})

###############################################################################
# Integer sort
###############################################################################
# Define ressources
SET(SRC_INT
integer_default/PVIntegerDefaultSortingFunc.cpp
integer_default/register.cpp
)

#Create lib
add_library(sorting_integer_default SHARED ${SRC_INT})
target_link_libraries(sorting_integer_default ${LINK_LIBRARIES})

###############################################################################
# IPV4 Sort
###############################################################################
# Define ressources
SET(SRC_IPV4
ipv4_default/PVIPv4DefaultSortingFunc.cpp
ipv4_default/register.cpp
)

#Create lib
add_library(sorting_ipv4_default SHARED ${SRC_IPV4})
target_link_libraries(sorting_ipv4_default ${LINK_LIBRARIES})

###############################################################################
# Install plugins and ressources
###############################################################################

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION sorting-functions
                                     RUNTIME DESTINATION sorting-functions)

INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION sorting-functions)
