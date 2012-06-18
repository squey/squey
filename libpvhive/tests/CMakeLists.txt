
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})

set(COMMON_LIBRARIES ${PVHIVE_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY})

add_executable(Tsimplistic simplistic.cpp)
target_link_libraries(Tsimplistic ${COMMON_LIBRARIES})

add_executable(Thive_thread_safe thread_safe.cpp)
target_link_libraries(Thive_thread_safe ${COMMON_LIBRARIES})

QT4_WRAP_CPP(TEST_ADRIEN_MOC_SRC test_adrien_dlg.h test_adrien_hdr.h OPTIONS ${QT4_WRAP_OPTIONS})
add_executable(Ttest_adrien-boost_thread test_adrien_main-boost_thread.cpp ${TEST_ADRIEN_MOC_SRC})
target_link_libraries(Ttest_adrien-boost_thread ${COMMON_LIBRARIES})

add_executable(Ttest_adrien-qt_thread test_adrien_main-qt_thread.cpp ${TEST_ADRIEN_MOC_SRC})
target_link_libraries(Ttest_adrien-qt_thread ${COMMON_LIBRARIES})
