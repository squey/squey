#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

#
# Mapping filters
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

include_directories(AFTER ${ICU_INCLUDE})

set(FILTERS_DIST_LIST
mapping_filter_float_default
mapping_filter_float_fraction
mapping_filter_integer_default
mapping_filter_ipv4_default
mapping_filter_enum_default
mapping_filter_host_default
mapping_filter_string_default
mapping_filter_string_4Bsort
mapping_filter_string_sort
mapping_filter_time_24h
mapping_filter_time_default
mapping_filter_userdefined_default
mapping_filter_mandatory_minmax
)

set(FILTERS_DIST_FILES
)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

add_library(mapping_filter_float_default SHARED float_default/PVMappingFilterFloatDefault.cpp float_default/register.cpp)
target_link_libraries(mapping_filter_float_default ${LINK_LIBRARIES})

add_library(mapping_filter_float_fraction SHARED float_fraction/PVMappingFilterFloatFraction.cpp float_fraction/register.cpp)
target_link_libraries(mapping_filter_float_fraction ${LINK_LIBRARIES})

add_library(mapping_filter_integer_default SHARED integer_default/PVMappingFilterIntegerDefault.cpp integer_default/register.cpp)
target_link_libraries(mapping_filter_integer_default ${LINK_LIBRARIES})

add_library(mapping_filter_ipv4_default SHARED ipv4_default/PVMappingFilterIPv4Default.cpp ipv4_default/register.cpp)
target_link_libraries(mapping_filter_ipv4_default ${LINK_LIBRARIES})

add_library(mapping_filter_enum_default SHARED enum_default/PVMappingFilterEnumDefault.cpp enum_default/register.cpp)
target_link_libraries(mapping_filter_enum_default ${LINK_LIBRARIES})

add_library(mapping_filter_host_default SHARED host_default/PVMappingFilterHostDefault.cpp host_default/register.cpp)
target_link_libraries(mapping_filter_host_default ${LINK_LIBRARIES})

add_library(mapping_filter_string_default SHARED string_default/PVMappingFilterStringDefault.cpp string_default/register.cpp)
target_link_libraries(mapping_filter_string_default ${LINK_LIBRARIES} ${TBB_LIBRARIES})

add_library(mapping_filter_string_4Bsort SHARED string_4Bsort/PVMappingFilterString4Bsort.cpp string_4Bsort/register.cpp)
target_link_libraries(mapping_filter_string_4Bsort ${LINK_LIBRARIES} ${TBB_LIBRARIES})

add_library(mapping_filter_string_sort SHARED string_sort/PVMappingFilterStringSort.cpp string_sort/register.cpp)
target_link_libraries(mapping_filter_string_sort ${LINK_LIBRARIES})

add_library(mapping_filter_time_24h SHARED time/24h/PVMappingFilterTime24h.cpp time/default/PVMappingFilterTimeDefault.cpp time/24h/register.cpp)
target_link_libraries(mapping_filter_time_24h ${LINK_LIBRARIES} ${ICU_LIBRARY})

add_library(mapping_filter_time_week SHARED time/week/PVMappingFilterTimeWeek.cpp time/default/PVMappingFilterTimeDefault.cpp time/week/register.cpp)
target_link_libraries(mapping_filter_time_week ${LINK_LIBRARIES} ${ICU_LIBRARY})

add_library(mapping_filter_time_default SHARED time/default/PVMappingFilterTimeDefault.cpp time/default/register.cpp)
target_link_libraries(mapping_filter_time_default ${LINK_LIBRARIES} ${ICU_LIBRARY})

add_library(mapping_filter_userdefined_default SHARED user-defined_default/PVMappingFilterUserdefinedDefault.cpp user-defined_default/register.cpp)
target_link_libraries(mapping_filter_userdefined_default ${LINK_LIBRARIES})

add_library(mapping_filter_mandatory_minmax SHARED mandatory_minmax/PVMandatoryMappingFilterMinmax.cpp mandatory_minmax/register.cpp)
target_link_libraries(mapping_filter_mandatory_minmax ${LINK_LIBRARIES})


INSTALL(TARGETS ${FILTERS_DIST_LIST} LIBRARY DESTINATION mapping-filters
                                     RUNTIME DESTINATION mapping-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION mapping-filters)
