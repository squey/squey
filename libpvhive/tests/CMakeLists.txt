
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})

set(COMMON_LIBRARIES ${PVHIVE_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY})

add_executable(Thive_test hive_test.cpp)
target_link_libraries(Thive_test ${COMMON_LIBRARIES})

add_executable(Thive_thread_safe thread_safe.cpp)
target_link_libraries(Thive_thread_safe ${COMMON_LIBRARIES})
