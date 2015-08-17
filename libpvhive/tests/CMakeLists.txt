#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

configure_file(run-massive-call-obj.sh.cmake run-massive-call-obj.sh @ONLY)
configure_file(run-massive-call-prop.sh.cmake run-massive-call-prop.sh @ONLY)
configure_file(run-massive-register.sh.cmake run-massive-register.sh @ONLY)


set(COMMON_LIBRARIES pvhive pvkernel picviz ${TBB_LIBRARIES} ${BOOST_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY} ${QT_LIBRARIES})

declare_pv_test(Tsimplistic simplistic.cpp)
target_link_libraries(Tsimplistic ${COMMON_LIBRARIES})

declare_pv_test(Tproperties properties.cpp)
target_link_libraries(Tproperties ${COMMON_LIBRARIES})

declare_pv_test(Tthread_safe thread_safe.cpp)
target_link_libraries(Tthread_safe ${COMMON_LIBRARIES})

QT5_WRAP_CPP(TEST_ADRIEN_MOC_SRC adrien_dlg.h adrien_hdr.h OPTIONS ${QT5_WRAP_OPTIONS})

declare_pv_test(Tadrien-boost_thread adrien_main-boost_thread.cpp ${TEST_ADRIEN_MOC_SRC})
target_link_libraries(Tadrien-boost_thread ${COMMON_LIBRARIES})

declare_pv_test(Tadrien-qt_thread adrien_main-qt_thread.cpp ${TEST_ADRIEN_MOC_SRC})
target_link_libraries(Tadrien-qt_thread ${COMMON_LIBRARIES})

QT5_WRAP_CPP(TEST_FUNCTIONAL_MOC_SRC functional_dlg.h functional_objs.h OPTIONS ${QT5_WRAP_OPTIONS})

declare_pv_test(Tfunctional functional_main.cpp functional_objs.cpp ${TEST_FUNCTIONAL_MOC_SRC})
target_link_libraries(Tfunctional ${COMMON_LIBRARIES})

declare_pv_test(Tinter-threads inter-threads.cpp)
target_link_libraries(Tinter-threads ${COMMON_LIBRARIES})

QT5_WRAP_CPP(TEST_DIT_MOC_SRC data-in-thread_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})

declare_pv_test(Tdata-in-thread data-in-thread.cpp ${TEST_DIT_MOC_SRC})
target_link_libraries(Tdata-in-thread ${COMMON_LIBRARIES})

QT5_WRAP_CPP(TEST_MODAL_MOC_SRC modal_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})

declare_pv_test(Tmodal modal.cpp ${TEST_MODAL_MOC_SRC})
target_link_libraries(Tmodal ${COMMON_LIBRARIES})

declare_pv_test(Tmultithreads multithreads.cpp)
target_link_libraries(Tmultithreads ${COMMON_LIBRARIES})

declare_pv_test(Thive_massive_mem massive-mem.cpp massive_common.cpp)
target_link_libraries(Thive_massive_mem ${COMMON_LIBRARIES})
add_test(TestHiveMemory Thive_massive_mem 100000)

declare_pv_test(Thive_massive_register massive-register.cpp massive_common.cpp)
target_link_libraries(Thive_massive_register ${COMMON_LIBRARIES})
add_test(TestHiveRegister run-massive-register.sh)

declare_pv_test(Thive_massive_call_obj massive-call-obj.cpp massive_common.cpp)
target_link_libraries(Thive_massive_call_obj ${COMMON_LIBRARIES})
add_test(TestHiveCallObject run-massive-call-obj.sh)

declare_pv_test(Thive_massive_call_prop massive-call-prop.cpp massive_common.cpp)
target_link_libraries(Thive_massive_call_prop ${COMMON_LIBRARIES})
add_test(TestHiveCallProperty run-massive-call-prop.sh)

QT5_WRAP_CPP(TEST_MQ_MOC_SRC message-queue.h OPTIONS ${QT5_WRAP_OPTIONS})

declare_pv_test(Tmessage-queue message-queue.cpp ${TEST_MQ_MOC_SRC})
target_link_libraries(Tmessage-queue ${COMMON_LIBRARIES})

include_directories(AFTER ${PICVIZ_INCLUDE_DIR})

declare_pv_test(Thive_call_helper call_helper.cpp)
target_link_libraries(Thive_call_helper ${COMMON_LIBRARIES})
add_test(TestHiveCallHelper Thive_call_helper)

QT5_WRAP_CPP(TEST_FUNC_OBSERVER_THREAD_MOC_SRC func_observer_thread.h OPTIONS ${QT5_WRAP_OPTIONS})
declare_pv_test(Tcore_func_observer_thread func_observer_thread.cpp ${TEST_FUNC_OBSERVER_THREAD_MOC_SRC})
target_link_libraries(Tcore_func_observer_thread ${COMMON_LIBRARIES})

declare_lib_tests_target()
