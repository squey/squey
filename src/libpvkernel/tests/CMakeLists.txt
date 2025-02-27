#
# @file
#

set(LINK_LIBRARIES pvkernel)
set(TESTS_FILES_DIR "../${TESTS_FILES_DIR}")
add_definitions(-fopenmp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

add_subdirectory(core)
add_subdirectory(rush)
add_subdirectory(widgets)
