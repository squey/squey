#
# Functions
#

include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)

# PCRE
include_directories(AFTER ${PCRE_INCLUDE_DIR})
# PCAP
include_directories(AFTER ${PCAP_INCLUDE_DIRS})

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/)

set(FUNCTIONS_DIST_LIST
function_mandatory_mapping_minmax
function_mapping_enum_default
# function_mapping_enum10_default
function_mapping_float_default
function_mapping_integer_default
#function_mapping_integer_log10
function_mapping_ipv4_default
#function_mapping_ipv4hexa_default
# function_mapping_string_ascii_val
function_mapping_string_default
function_mapping_time_default
function_mapping_time_24h
# function_mapping_time_week
# function_mapping_time_month
#function_mapping_time_2021ct
# function_mapping_enumint_default
# function_mapping_enumipv4_default
function_mapping_user-defined_default

function_plotting_float_default
function_plotting_float_minmax
function_plotting_time_default
function_plotting_time_minmax
function_plotting_enum_default
# function_plotting_enum10_default
function_plotting_integer_default
function_plotting_integer_minmax
function_plotting_integer_port
function_plotting_ipv4_default
# function_plotting_ipv4hexa_default
function_plotting_string_default
# function_plotting_enumint_default
# function_plotting_enumipv4_default
function_plotting_user-defined_default
)

set(LINK_LIBRARIES picviz ${PVCORE_LIBRARY} ${QT_QTCORE_LIBRARY})


#
# Mandatory functions
#
add_library(function_mandatory_mapping_minmax SHARED mandatory/minmax.cpp)
target_link_libraries(function_mandatory_mapping_minmax ${LINK_LIBRARIES})

#
# Mapping functions
#
add_library(function_mapping_enum_default SHARED mapping/enum_default.cpp)
target_link_libraries(function_mapping_enum_default ${LINK_LIBRARIES})
# add_library(function_mapping_enum10_default SHARED mapping/enum10_default.cpp)
# target_link_libraries(function_mapping_enum10_default ${LINK_LIBRARIES})
add_library(function_mapping_float_default SHARED mapping/float_default.cpp)
target_link_libraries(function_mapping_float_default ${LINK_LIBRARIES})
add_library(function_mapping_integer_default SHARED mapping/integer_default.cpp)
target_link_libraries(function_mapping_integer_default ${LINK_LIBRARIES})
#add_library(function_mapping_integer_log10 SHARED mapping/integer_log10.cpp)
#target_link_libraries(function_mapping_integer_log10 ${LINK_LIBRARIES})
add_library(function_mapping_ipv4_default SHARED mapping/ipv4_default.cpp)
target_link_libraries(function_mapping_ipv4_default ${LINK_LIBRARIES})
#add_library(function_mapping_ipv4hexa_default SHARED mapping/ipv4hexa_default.cpp)
#target_link_libraries(function_mapping_ipv4hexa_default ${LINK_LIBRARIES})
# add_library(function_mapping_string_ascii_val SHARED mapping/string_ascii_val.cpp)
# target_link_libraries(function_mapping_string_ascii_val ${LINK_LIBRARIES})
add_library(function_mapping_string_default SHARED mapping/string_default.cpp)
target_link_libraries(function_mapping_string_default ${LINK_LIBRARIES})

add_library(function_mapping_time_default SHARED mapping/time_default.cpp mapping/time/core.cpp)
target_link_libraries(function_mapping_time_default ${LINK_LIBRARIES})
add_library(function_mapping_time_24h SHARED mapping/time_24h.cpp mapping/time/core.cpp)
target_link_libraries(function_mapping_time_24h ${LINK_LIBRARIES})
# add_library(function_mapping_time_week SHARED mapping/time_week.cpp)
# target_link_libraries(function_mapping_time_week ${LINK_LIBRARIES})
# add_library(function_mapping_time_month SHARED mapping/time_month.cpp)
# target_link_libraries(function_mapping_time_month ${LINK_LIBRARIES})
# add_library(function_mapping_time_2021ct SHARED mapping/time_2021ct.cpp)
# target_link_libraries(function_mapping_time_2021ct ${LINK_LIBRARIES})
# add_library(function_mapping_enumint_default SHARED mapping/enumint_default.cpp)
# target_link_libraries(function_mapping_enumint_default ${LINK_LIBRARIES})
# add_library(function_mapping_enumipv4_default SHARED mapping/enumipv4_default.cpp)
# target_link_libraries(function_mapping_enumipv4_default ${LINK_LIBRARIES})
add_library(function_mapping_user-defined_default SHARED mapping/user-defined_default.cpp)
target_link_libraries(function_mapping_user-defined_default ${LINK_LIBRARIES})


#
# Plotting functions
#
add_library(function_plotting_float_default SHARED plotting/float_default.cpp)
target_link_libraries(function_plotting_float_default ${LINK_LIBRARIES})
add_library(function_plotting_float_minmax SHARED plotting/float_minmax.cpp)
target_link_libraries(function_plotting_float_minmax ${LINK_LIBRARIES})
add_library(function_plotting_time_default SHARED plotting/time_default.cpp)
target_link_libraries(function_plotting_time_default ${LINK_LIBRARIES})
add_library(function_plotting_time_minmax SHARED plotting/time_minmax.cpp)
target_link_libraries(function_plotting_time_minmax ${LINK_LIBRARIES})
add_library(function_plotting_enum_default SHARED plotting/enum_default.cpp)
target_link_libraries(function_plotting_enum_default ${LINK_LIBRARIES})
# add_library(function_plotting_enum10_default SHARED plotting/enum10_default.cpp)
# target_link_libraries(function_plotting_enum10_default ${LINK_LIBRARIES})
add_library(function_plotting_integer_default SHARED plotting/integer_default.cpp)
target_link_libraries(function_plotting_integer_default ${LINK_LIBRARIES})
add_library(function_plotting_integer_minmax SHARED plotting/integer_minmax.cpp)
target_link_libraries(function_plotting_integer_minmax ${LINK_LIBRARIES})
add_library(function_plotting_integer_port SHARED plotting/integer_port.cpp)
target_link_libraries(function_plotting_integer_port ${LINK_LIBRARIES})
add_library(function_plotting_ipv4_default SHARED plotting/ipv4_default.cpp)
target_link_libraries(function_plotting_ipv4_default ${LINK_LIBRARIES})
# add_library(function_plotting_ipv4hexa_default SHARED plotting/ipv4hexa_default.cpp)
# target_link_libraries(function_plotting_ipv4hexa_default ${LINK_LIBRARIES})
add_library(function_plotting_string_default SHARED plotting/string_default.cpp)
target_link_libraries(function_plotting_string_default ${LINK_LIBRARIES})
# add_library(function_plotting_enumint_default SHARED plotting/enumint_default.cpp)
# target_link_libraries(function_plotting_enumint_default ${LINK_LIBRARIES})
# add_library(function_plotting_enumipv4_default SHARED plotting/enumipv4_default.cpp)
# target_link_libraries(function_plotting_enumipv4_default ${LINK_LIBRARIES})
add_library(function_plotting_user-defined_default SHARED plotting/user-defined_default.cpp)
target_link_libraries(function_plotting_user-defined_default ${LINK_LIBRARIES})

INSTALL(TARGETS ${FUNCTIONS_DIST_LIST} DESTINATION functions)
