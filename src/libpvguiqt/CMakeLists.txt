#
# @file
#

project(libpvguiqt)

add_compile_definitions(BUILD_PVGUIQT)

include(CTest)
set(CTEST_PROJECT_NAME "PVGuiQT")

set(COPYING_DIR_NAME "COPYING")
add_compile_definitions(SQUEY_COPYING_DIR="${SHARE_RESSOURCES}/${COPYING_DIR_NAME}")
install(DIRECTORY ${squey_SOURCE_DIR}/${COPYING_DIR_NAME} DESTINATION ${SHARE_RESSOURCES})

# For KSyntaxHighlighting
find_package(KF6SyntaxHighlighting REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)
