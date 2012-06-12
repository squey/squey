
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})

add_executable(Thive_test hive_test.cpp)
target_link_libraries(Thive_test ${PVHIVE_LIBRARIES})
