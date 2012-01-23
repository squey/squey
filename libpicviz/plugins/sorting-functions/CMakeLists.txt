#
# Layer filters
#
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

link_directories(${picviz-inspector_SOURCE_DIR}/libpicviz/src/)

set(FUNCTIONS_DIST_LIST
sorting_float_default
sorting_integer_default
)

set(FUNCTIONS_DIST_FILES
)

set(LINK_LIBRARIES ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARIES} ${QT_LIBRARIES})

add_library(sorting_float_default SHARED float_default/PVFloatDefaultSortingFunc.cpp float_default/register.cpp)
target_link_libraries(sorting_float_default ${LINK_LIBRARIES})

add_library(sorting_integer_default SHARED integer_default/PVIntegerDefaultSortingFunc.cpp integer_default/register.cpp)
target_link_libraries(sorting_integer_default ${LINK_LIBRARIES})

INSTALL(TARGETS ${FUNCTIONS_DIST_LIST} LIBRARY DESTINATION sorting-functions
                                     RUNTIME DESTINATION sorting-functions)

INSTALL(FILES ${FUNCTIONS_DIST_FILES} DESTINATION sorting-functions)
