include(CTest)
set(CTEST_PROJECT_NAME "PVHive")

include_directories(AFTER ${PVHIVE_INCLUDE_DIRS})
include_directories(AFTER ${Boost_INCLUDE_DIR})

add_definitions(-DDEBUG)

add_definitions(${COMPILER_OPTIMIZE_FLAGS})

add_subdirectory(src)
add_subdirectory(tests)

