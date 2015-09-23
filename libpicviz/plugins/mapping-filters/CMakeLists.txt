#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

###############################################################################
# Mapping filters
###############################################################################

###############################################################################
# Define mapping filters plugins to distribute
###############################################################################

set(FILTERS_DIST_LIST
mapping_filter_float_default
mapping_filter_float_fraction
mapping_filter_integer_default
mapping_filter_integer_hex
mapping_filter_integer_oct
mapping_filter_ipv4_default
mapping_filter_ipv4_uniform
mapping_filter_enum_default
mapping_filter_host_default
mapping_filter_string_default
mapping_filter_string_4Bsort
mapping_filter_time_24h
mapping_filter_time_default
mapping_filter_time_week
mapping_filter_mandatory_minmax
)

set(FILTERS_DIST_FILES
)

# Linkage libs
set(LINK_LIBRARIES picviz pvkernel)

###############################################################################
# Float mapping filters
###############################################################################
#Define ressources
SET(SRC_FLOAT
float_default/PVMappingFilterFloatDefault.cpp
float_default/register.cpp)

#Create lib
add_library(mapping_filter_float_default SHARED ${SRC_FLOAT})
target_link_libraries(mapping_filter_float_default ${LINK_LIBRARIES})

###############################################################################
# Float fraction mapping filters
###############################################################################
#Define ressources
SET(SRC_FRACTION
float_fraction/PVMappingFilterFloatFraction.cpp
float_fraction/register.cpp)

#Create lib
add_library(mapping_filter_float_fraction SHARED ${SRC_FRACTION})
target_link_libraries(mapping_filter_float_fraction ${LINK_LIBRARIES})

###############################################################################
# Integer mapping filters
###############################################################################
#Define ressources
SET(SRC_INTEGER
integer_default/PVMappingFilterIntegerDefault.cpp
integer_default/register.cpp)

#Create lib
add_library(mapping_filter_integer_default SHARED ${SRC_INTEGER})
target_link_libraries(mapping_filter_integer_default ${LINK_LIBRARIES})

###############################################################################
# Hexadecimal mapping filters
###############################################################################
#Define ressources
SET(SRC_HEXA
integer_hex/PVMappingFilterIntegerHex.cpp
integer_hex/register.cpp)

#Create lib
add_library(mapping_filter_integer_hex SHARED ${SRC_HEXA})
target_link_libraries(mapping_filter_integer_hex ${LINK_LIBRARIES})

###############################################################################
# Octadecimal mapping filters
###############################################################################
#Define ressources
SET(SRC_OCT
integer_oct/PVMappingFilterIntegerOct.cpp
integer_oct/register.cpp)

#Create lib
add_library(mapping_filter_integer_oct SHARED ${SRC_OCT})
target_link_libraries(mapping_filter_integer_oct ${LINK_LIBRARIES})

###############################################################################
# IPV4 mapping filters
###############################################################################
#Define ressources
SET(SRC_IPV4
ipv4_default/PVMappingFilterIPv4Default.cpp
ipv4_default/register.cpp)

#Create lib
add_library(mapping_filter_ipv4_default SHARED ${SRC_IPV4})
target_link_libraries(mapping_filter_ipv4_default ${LINK_LIBRARIES})

###############################################################################
# IPV4 uniform mapping filters
###############################################################################
#Define ressources
SET(SRC_IPV4_UNIFORM
ipv4_uniform/PVMappingFilterIPv4Uniform.cpp
ipv4_uniform/register.cpp)

#Create lib
add_library(mapping_filter_ipv4_uniform SHARED ${SRC_IPV4_UNIFORM})
target_link_libraries(mapping_filter_ipv4_uniform ${LINK_LIBRARIES})

###############################################################################
# Enum mapping filters
###############################################################################
#Define ressources
SET(SRC_ENUM
enum_default/PVMappingFilterEnumDefault.cpp
enum_default/register.cpp)

#Create lib
add_library(mapping_filter_enum_default SHARED ${SRC_ENUM})
target_link_libraries(mapping_filter_enum_default ${LINK_LIBRARIES})

###############################################################################
# Host mapping filters
###############################################################################
#Define ressources
SET(SRC_HOST
host_default/PVMappingFilterHostDefault.cpp
host_default/register.cpp)

#Create lib
add_library(mapping_filter_host_default SHARED ${SRC_HOST})
target_link_libraries(mapping_filter_host_default ${LINK_LIBRARIES})

###############################################################################
# String mapping filters
###############################################################################
#Define ressources
SET(SRC_STRING
string_default/PVMappingFilterStringDefault.cpp
string_default/register.cpp)

#Create lib
add_library(mapping_filter_string_default SHARED ${SRC_STRING})
target_link_libraries(mapping_filter_string_default ${LINK_LIBRARIES} ${TBB_LIBRARIES})

###############################################################################
# String 4B mapping filters
###############################################################################
#Define ressources
SET(SRC_STRING4B
string_4Bsort/PVMappingFilterString4Bsort.cpp
string_4Bsort/register.cpp)

#Create lib
add_library(mapping_filter_string_4Bsort SHARED ${SRC_STRING4B})
target_link_libraries(mapping_filter_string_4Bsort ${LINK_LIBRARIES} ${TBB_LIBRARIES})

###############################################################################
# Time 24H mapping filters
###############################################################################
#Define ressources
SET(SRC_TIME24
time/24h/PVMappingFilterTime24h.cpp
time/default/PVMappingFilterTimeDefault.cpp time/24h/register.cpp)

#Create lib
add_library(mapping_filter_time_24h SHARED ${SRC_TIME24})
target_link_libraries(mapping_filter_time_24h ${LINK_LIBRARIES} ${ICU_LIBRARY})

###############################################################################
# Time week mapping filters
###############################################################################
#Define ressources
SET(SRC_TIMEWEEK
time/week/PVMappingFilterTimeWeek.cpp
time/default/PVMappingFilterTimeDefault.cpp time/week/register.cpp)

#Create lib
add_library(mapping_filter_time_week SHARED ${SRC_TIMEWEEK})
target_link_libraries(mapping_filter_time_week ${LINK_LIBRARIES} ${ICU_LIBRARY})

###############################################################################
# Time mapping filters
###############################################################################
#Define ressources
SET(SRC_TIME
time/default/PVMappingFilterTimeDefault.cpp
time/default/register.cpp)

#Create lib
add_library(mapping_filter_time_default SHARED ${SRC_TIME})
target_link_libraries(mapping_filter_time_default ${LINK_LIBRARIES} ${ICU_LIBRARY})

###############################################################################
# min max mapping filters
###############################################################################
#Define ressources
SET(SRC_MINMAX
mandatory_minmax/PVMandatoryMappingFilterMinmax.cpp
mandatory_minmax/register.cpp)

#Create lib
add_library(mapping_filter_mandatory_minmax SHARED ${SRC_MINMAX})
target_link_libraries(mapping_filter_mandatory_minmax ${LINK_LIBRARIES})

###############################################################################
# Date month mapping filters
###############################################################################
#Define ressources
SET(SRC_DATEMONTH
date/month/PVMappingFilterDateMonth.cpp
date/month/register.cpp date/PVMappingFilterDateBase.cpp)

#Create lib
add_library(mapping_filter_date_month SHARED ${SRC_DATEMONTH})
target_link_libraries(mapping_filter_date_month ${LINK_LIBRARIES} ${ICU_LIBRARY})

###############################################################################
# Date day mapping filters
###############################################################################
#Define ressources
SET(SRC_DATEDAY
date/day/PVMappingFilterDateDay.cpp
date/day/register.cpp date/PVMappingFilterDateBase.cpp)

#Create lib
add_library(mapping_filter_date_day SHARED ${SRC_DATEDAY})
target_link_libraries(mapping_filter_date_day ${LINK_LIBRARIES} ${ICU_LIBRARY})

###############################################################################
# Define install directory
###############################################################################

INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION mapping-filters
                                     RUNTIME DESTINATION mapping-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION mapping-filters)
