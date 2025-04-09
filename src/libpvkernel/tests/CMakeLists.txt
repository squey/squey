#
# @file
#

set(LINK_LIBRARIES pvkernel)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TESTS_FILES_DIR "../${TESTS_FILES_DIR}")
endif()
add_definitions(-fopenmp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

add_subdirectory(core)
add_subdirectory(rush)
add_subdirectory(widgets)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)