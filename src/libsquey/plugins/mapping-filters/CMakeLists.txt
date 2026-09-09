#
# @file
#

###############################################################################
# Mapping filters
###############################################################################

###############################################################################
# Define mapping filters plugins to distribute
###############################################################################

set(FILTERS_DIST_LIST
mapping_filter_default
mapping_filter_4bsort
mapping_filter_host
mapping_filter_string
mapping_filter_24h
mapping_filter_week
mapping_filter_mac_address
)

# Linkage libs
set(LINK_LIBRARIES pvsquey pvkernel)

###############################################################################
# Default mapping filters
###############################################################################
#Define ressources
set(SRC_DEFAULT
default/PVMappingFilterDefault.cpp
default/register.cpp)

#Create lib
add_library(mapping_filter_default SHARED ${SRC_DEFAULT})
target_link_libraries(mapping_filter_default PRIVATE ${LINK_LIBRARIES})

###############################################################################
# 4BSort mapping filters
###############################################################################
#Define ressources
set(SRC_4BSORT
4Bsort/PVMappingFilter4Bsort.cpp
4Bsort/register.cpp)

#Create lib
add_library(mapping_filter_4bsort SHARED ${SRC_4BSORT})
target_link_libraries(mapping_filter_4bsort PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Host mapping filters
###############################################################################
#Define ressources
set(SRC_HOST
host/PVMappingFilterHost.cpp
host/register.cpp)

#Create lib
add_library(mapping_filter_host SHARED ${SRC_HOST})
target_link_libraries(mapping_filter_host PRIVATE ${LINK_LIBRARIES})

###############################################################################
# String mapping filters
###############################################################################
#Define ressources
set(SRC_STRING
string/PVMappingFilterString.cpp
string/register.cpp)

#Create lib
add_library(mapping_filter_string SHARED ${SRC_STRING})
target_link_libraries(mapping_filter_string PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Time 24H mapping filters
###############################################################################
#Define ressources
set(SRC_TIME24
time24/PVMappingFilterTime24h.cpp
time24/register.cpp)

#Create lib
add_library(mapping_filter_24h SHARED ${SRC_TIME24})
target_link_libraries(mapping_filter_24h PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Time week mapping filters
###############################################################################
#Define ressources
set(SRC_TIMEWEEK
timeweek/PVMappingFilterTimeWeek.cpp
timeweek/register.cpp)

#Create lib
add_library(mapping_filter_week SHARED ${SRC_TIMEWEEK})
target_link_libraries(mapping_filter_week PRIVATE ${LINK_LIBRARIES})

###############################################################################
# MAC address mapping filters
###############################################################################
#Define ressources
set(SRC_MACADDRESS
mac_address/PVMappingFilterMacAddress.cpp
mac_address/register.cpp)

#Create lib
add_library(mapping_filter_mac_address SHARED ${SRC_MACADDRESS})
target_link_libraries(mapping_filter_mac_address PRIVATE ${LINK_LIBRARIES})

###############################################################################
# Define install directory
###############################################################################

install(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION ${SQUEY_PLUGIN}/mapping-filters
                                     RUNTIME DESTINATION ${SQUEY_PLUGIN}/mapping-filters)
