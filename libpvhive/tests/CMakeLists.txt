#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

################################################
# QT5 related include and check
################################################

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

set(COMMON_LIBRARIES pvhive pvkernel picviz ${TBB_LIBRARIES} ${BOOST_LIBRARIES}
    ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY})

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
add_test(TestHiveMemory Thive_massive_mem 1000)

###############################################################################
# Massive register checking 
###############################################################################

declare_pv_test(Thive_massive_register massive-register.cpp massive_common.cpp)
target_link_libraries(Thive_massive_register ${COMMON_LIBRARIES})

# object registration
add_test(TestHiveRegister ./Thive_massive_register 1000 0 0 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 10000 0 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 100000 0 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1000000 0 0 0 0 0)
endif()

# property registration
# varying property number
add_test(TestHiveRegister ./Thive_massive_register 1 1000 0 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 10000 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 100000 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 1000000 0 0 0 0)
endif()

# varying objects/properties ratio
add_test(TestHiveRegister ./Thive_massive_register 10 1000 0 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 10 100000 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 100 10000 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1000 1000 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 10000 100 0 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 100000 10 0 0 0 0)
endif()

# object actors registration
# varying actors number
add_test(TestHiveRegister ./Thive_massive_register 1 0 1000 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 0 10000 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 0 100000 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 0 1000000 0 0 0)
endif()

# varying objects/actors ratio
add_test(TestHiveRegister ./Thive_massive_register 10 0 1000 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 10 0 100000 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 100 0 10000 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1000 0 1000 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 10000 0 100 0 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 100000 0 10 0 0 0)
endif()

# property actors registration
# varying actors number
add_test(TestHiveRegister ./Thive_massive_register 1 1 0 1000 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 1 0 10000 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 1 0 100000 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 1 0 1000000 0 0)
endif()

# varying properties/actors ratio
add_test(TestHiveRegister ./Thive_massive_register 1 10 0 1000 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 10 0 100000 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 100 0 10000 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 1000 0 1000 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 10000 0 100 0 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 100000 0 10 0 0)
endif()
  
# object pbservers registration
# varying observers number
add_test(TestHiveRegister ./Thive_massive_register 1 0 0 0 1000 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 0 0 0 10000 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 0 0 0 100000 0)
    add_test(TestHiveRegister ./Thive_massive_register 1 0 0 0 1000000 0)
endif()
     
# varying objects/observers ratio
add_test(TestHiveRegister ./Thive_massive_register 10 0 0 0 1000 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 10 0 0 0 100000 0)
    add_test(TestHiveRegister ./Thive_massive_register 100 0 0 0 10000 0)
    add_test(TestHiveRegister ./Thive_massive_register 1000 0 0 0 1000 0)
    add_test(TestHiveRegister ./Thive_massive_register 10000 0 0 0 100 0)
    add_test(TestHiveRegister ./Thive_massive_register 100000 0 0 0 10 0)
endif()
      
# property observers registration
# varying servers number
add_test(TestHiveRegister ./Thive_massive_register 1 1 0 0 0 1000)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 1 0 0 0 10000)
    add_test(TestHiveRegister ./Thive_massive_register 1 1 0 0 0 100000)
    add_test(TestHiveRegister ./Thive_massive_register 1 1 0 0 0 1000000)
endif()
 
# varying properties/observers ratio
add_test(TestHiveRegister ./Thive_massive_register 1 10 0 0 0 1000)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister ./Thive_massive_register 1 10 0 0 0 100000)
    add_test(TestHiveRegister ./Thive_massive_register 1 100 0 0 0 10000)
    add_test(TestHiveRegister ./Thive_massive_register 1 1000 0 0 0 1000)
    add_test(TestHiveRegister ./Thive_massive_register 1 10000 0 0 0 100)
    add_test(TestHiveRegister ./Thive_massive_register 1 100000 0 0 0 10)
endif()


###############################################################################
# Massive call checking 
# * Check done for call properties and call object
###############################################################################

macro(PV_MASSIVE_TEST acces)
    declare_pv_test(Thive_massive_call_${acces} massive-call-${acces}.cpp massive_common.cpp)
    target_link_libraries(Thive_massive_call_${acces} ${COMMON_LIBRARIES})
    add_test(TestHiveCallObject Thive_massive_call_${acces} 100 100 100)

    if(DEFINED TESTS_LONG)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 10000000 1       0)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 10000    1000    0)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 10       1000000 0)

	add_test(TestHiveCallObject Thive_massive_call_${acces} 10000000 1       1)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 10000    1000    1)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 10       1000000 1)

	add_test(TestHiveCallObject Thive_massive_call_${acces} 1000000 1       1000)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 1000    1000    1000)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 1       1000000 1000)

	add_test(TestHiveCallObject Thive_massive_call_${acces} 10000000 1    1000000)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 1000    1    1000000)
	add_test(TestHiveCallObject Thive_massive_call_${acces} 1       1000 1000000)
    endif()
endmacro()

PV_MASSIVE_TEST(obj)
PV_MASSIVE_TEST(prop)

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
