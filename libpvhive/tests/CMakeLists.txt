#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

################################################
# QT5 related include and check
################################################

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(AFTER ${INENDI_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(COMMON_LIBRARIES pvhive pvkernel inendi ${TBB_LIBRARIES} ${BOOST_LIBRARIES}
    ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY})

add_pv_compile_test(Tsimplistic LINKS ${COMMON_LIBRARIES} FILES simplistic.cpp)
add_pv_compile_test(Tthread_safe LINKS ${COMMON_LIBRARIES} FILES thread_safe.cpp)

add_pv_compile_test(Tinter-threads LINKS ${COMMON_LIBRARIES} FILES inter-threads.cpp)

QT5_WRAP_CPP(TEST_DIT_MOC_SRC data-in-thread_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tdata-in-thread  LINKS ${COMMON_LIBRARIES} FILES data-in-thread.cpp ${TEST_DIT_MOC_SRC})

QT5_WRAP_CPP(TEST_MODAL_MOC_SRC modal_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tmodal LINKS ${COMMON_LIBRARIES} FILES modal.cpp ${TEST_MODAL_MOC_SRC})
add_pv_compile_test(Tmultithreads LINKS ${COMMON_LIBRARIES} FILES multithreads.cpp)

QT5_WRAP_CPP(TEST_MQ_MOC_SRC message-queue.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tmessage-queue LINKS ${COMMON_LIBRARIES} FILES message-queue.cpp ${TEST_MQ_MOC_SRC})
add_pv_compile_test(Thive_call_helper LINKS ${COMMON_LIBRARIES} FILES call_helper.cpp)
add_test(Thive_call_helper Thive_call_helper)

QT5_WRAP_CPP(TEST_FUNC_OBSERVER_THREAD_MOC_SRC func_observer_thread.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tcore_func_observer_thread LINKS ${COMMON_LIBRARIES} FILES func_observer_thread.cpp ${TEST_FUNC_OBSERVER_THREAD_MOC_SRC})
