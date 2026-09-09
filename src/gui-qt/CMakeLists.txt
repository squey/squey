#
# @file
#

project(gui-qt)

add_compile_definitions(BUILD_SQUEY)

include(CTest)
set(CTEST_PROJECT_NAME "GuiQt")

# ui_*.h files generated beside the sources
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(src)
add_subdirectory(tests)

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)
