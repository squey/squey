#
# Mapping filters
#
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)

include_directories(AFTER ${ICU_INCLUDE})

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpicviz/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/)

set(FILTERS_DIST_LIST
mapping_filter_float_default
mapping_filter_integer_default
mapping_filter_ipv4_default
mapping_filter_enum_default
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

# 
# OpenMP
#
if (WIN32)
	add_definitions(/openmp)
else (WIN32)
	add_definitions(-fopenmp)
endif (WIN32)

add_library(mapping_filter_float_default SHARED float_default/PVMappingFilterFloatDefault.cpp float_default/register.cpp)
target_link_libraries(mapping_filter_float_default picviz pvfilter)

add_library(mapping_filter_integer_default SHARED integer_default/PVMappingFilterIntegerDefault.cpp integer_default/register.cpp)
target_link_libraries(mapping_filter_integer_default picviz pvfilter)

add_library(mapping_filter_ipv4_default SHARED ipv4_default/PVMappingFilterIPv4Default.cpp ipv4_default/register.cpp)
target_link_libraries(mapping_filter_ipv4_default picviz pvfilter)

add_library(mapping_filter_enum_default SHARED enum_default/PVMappingFilterEnumDefault.cpp enum_default/register.cpp)
target_link_libraries(mapping_filter_enum_default picviz pvfilter)

add_library(mapping_filter_host_default SHARED host_default/PVMappingFilterHostDefault.cpp host_default/register.cpp)
target_link_libraries(mapping_filter_host_default picviz pvfilter)

add_library(mapping_filter_string_default SHARED string_default/PVMappingFilterStringDefault.cpp string_default/register.cpp)
target_link_libraries(mapping_filter_string_default picviz pvfilter ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY})

add_library(mapping_filter_string_4Bsort SHARED string_4Bsort/PVMappingFilterString4Bsort.cpp string_4Bsort/register.cpp)
target_link_libraries(mapping_filter_string_4Bsort picviz pvfilter ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY})

add_library(mapping_filter_string_sort SHARED string_sort/PVMappingFilterStringSort.cpp string_sort/register.cpp)
target_link_libraries(mapping_filter_string_sort picviz pvfilter)

add_library(mapping_filter_time_24h SHARED time/24h/PVMappingFilterTime24h.cpp time/default/PVMappingFilterTimeDefault.cpp time/24h/register.cpp)
target_link_libraries(mapping_filter_time_24h picviz pvfilter ${ICU_LIBRARY})

add_library(mapping_filter_time_week SHARED time/week/PVMappingFilterTimeWeek.cpp time/default/PVMappingFilterTimeDefault.cpp time/week/register.cpp)
target_link_libraries(mapping_filter_time_week picviz pvfilter ${ICU_LIBRARY})

add_library(mapping_filter_time_default SHARED time/default/PVMappingFilterTimeDefault.cpp time/default/register.cpp)
target_link_libraries(mapping_filter_time_default picviz pvfilter ${ICU_LIBRARY})

add_library(mapping_filter_userdefined_default SHARED user-defined_default/PVMappingFilterUserdefinedDefault.cpp user-defined_default/register.cpp)
target_link_libraries(mapping_filter_userdefined_default picviz pvfilter)

add_library(mapping_filter_mandatory_minmax SHARED mandatory_minmax/PVMandatoryMappingFilterMinmax.cpp mandatory_minmax/register.cpp)
target_link_libraries(mapping_filter_mandatory_minmax picviz pvfilter)


INSTALL(TARGETS ${FILTERS_DIST_LIST} DESTINATION mapping-filters)
INSTALL(FILES ${FILTERS_DIST_FILES} DESTINATION mapping-filters)
