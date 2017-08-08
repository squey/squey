#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

set(LINK_LIBRARIES pvkernel ${TBB_LIBRARIES} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${ICU_LIBRARY} ${PVLOGGER_LIBRARIES})
add_definitions(-fopenmp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

add_subdirectory(core)
add_subdirectory(rush)
add_subdirectory(widgets)
