#
# @file
#

project(libpvparallelview)

# For KItemModels
find_package(KF6ItemModels REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)
