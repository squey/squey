#
# @file
#

###############################################################################
# Scaling filters
###############################################################################

###############################################################################
# Define scaling filter plugns to distribute
###############################################################################

set(FILTERS_DIST_LIST
scaling_filter_default
scaling_filter_log_minmax
scaling_filter_port
scaling_filter_enum
)

# Extra libs
set(LINK_LIBRARIES pvsquey pvkernel)

###############################################################################
# Default filter
###############################################################################
# Define ressources
SET(SRC_DEFAULT
default/PVScalingFilterMinmax.cpp
default/register.cpp
)

#Create lib
add_library(scaling_filter_default SHARED ${SRC_DEFAULT})
target_link_libraries(scaling_filter_default ${LINK_LIBRARIES})

# These two look at the values themselves to tell a NaN from a number, and a NaN
# is exactly what -ffast-math promises the compiler will never occur: under it,
# std::isnan is rejected outright by clang and even reading the bit pattern is
# folded away. The promise cannot be kept here -- the values come from files the
# user chose -- so it is withdrawn for these two objects only.
target_compile_options(scaling_filter_default PRIVATE -fno-fast-math)

###############################################################################
# Log min max filter
###############################################################################
# Define ressources
SET(SRC_LOGMINMAX
log_minmax/PVScalingFilterLogMinmax.cpp
log_minmax/register.cpp
)

#Create lib
add_library(scaling_filter_log_minmax SHARED ${SRC_LOGMINMAX})
target_link_libraries(scaling_filter_log_minmax ${LINK_LIBRARIES})
target_compile_options(scaling_filter_log_minmax PRIVATE -fno-fast-math)

###############################################################################
# Port filter
###############################################################################
# Define ressources
SET(SRC_PORT
port/PVScalingFilterPort.cpp
port/register.cpp
)

#Create lib
add_library(scaling_filter_port SHARED ${SRC_PORT})
target_link_libraries(scaling_filter_port ${LINK_LIBRARIES})

###############################################################################
# Enum filter
###############################################################################
# Define ressources
SET(SRC_ENUM
enum/PVScalingFilterEnum.cpp
enum/register.cpp
)

#Create lib
add_library(scaling_filter_enum SHARED ${SRC_ENUM})
target_link_libraries(scaling_filter_enum ${LINK_LIBRARIES})

###############################################################################
# Install plugins and ressources
###############################################################################
INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION ${SQUEY_PLUGIN}/scaling-filters
                                     RUNTIME DESTINATION ${SQUEY_PLUGIN}/scaling-filters)
