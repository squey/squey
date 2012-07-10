
include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVHIVE_INCLUDE_DIR})

set(COMMON_LIBRARIES ${PVHIVE_LIBRARIES} ${TBB_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY})

add_executable(Tsimplistic simplistic.cpp)
target_link_libraries(Tsimplistic ${COMMON_LIBRARIES})

add_executable(Tproperties properties.cpp)
target_link_libraries(Tproperties ${COMMON_LIBRARIES})

add_executable(Tthread_safe thread_safe.cpp)
target_link_libraries(Tthread_safe ${COMMON_LIBRARIES})

QT4_WRAP_CPP(TEST_ADRIEN_MOC_SRC adrien_dlg.h adrien_hdr.h OPTIONS ${QT4_WRAP_OPTIONS})

add_executable(Tadrien-boost_thread adrien_main-boost_thread.cpp ${TEST_ADRIEN_MOC_SRC})
target_link_libraries(Tadrien-boost_thread ${COMMON_LIBRARIES})

add_executable(Tadrien-qt_thread adrien_main-qt_thread.cpp ${TEST_ADRIEN_MOC_SRC})
target_link_libraries(Tadrien-qt_thread ${COMMON_LIBRARIES})

QT4_WRAP_CPP(TEST_FUNCTIONAL_MOC_SRC functional_dlg.h functional_objs.h OPTIONS ${QT4_WRAP_OPTIONS})

add_executable(Tfunctional functional_main.cpp functional_objs.cpp ${TEST_FUNCTIONAL_MOC_SRC})
target_link_libraries(Tfunctional ${COMMON_LIBRARIES} ${PVKERNEL_LIBRARIES})

add_executable(Tinter-threads inter-threads.cpp)
target_link_libraries(Tinter-threads ${COMMON_LIBRARIES} ${PVKERNEL_LIBRARIES})

QT4_WRAP_CPP(TEST_DIT_MOC_SRC data-in-thread_dlg.h OPTIONS ${QT4_WRAP_OPTIONS})

add_executable(Tdata-in-thread data-in-thread.cpp ${TEST_DIT_MOC_SRC})
target_link_libraries(Tdata-in-thread ${COMMON_LIBRARIES} ${PVKERNEL_LIBRARIES})

