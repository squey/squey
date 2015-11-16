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
add_pv_compile_test(Tproperties LINKS ${COMMON_LIBRARIES} FILES properties.cpp)
add_pv_compile_test(Tthread_safe LINKS ${COMMON_LIBRARIES} FILES thread_safe.cpp)

QT5_WRAP_CPP(TEST_ADRIEN_MOC_SRC adrien_dlg.h adrien_hdr.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tadrien-boost_thread LINKS ${COMMON_LIBRARIES} FILES adrien_main-boost_thread.cpp ${TEST_ADRIEN_MOC_SRC})
add_pv_compile_test(Tadrien-qt_thread LINKS ${COMMON_LIBRARIES} FILES adrien_main-qt_thread.cpp ${TEST_ADRIEN_MOC_SRC})

QT5_WRAP_CPP(TEST_FUNCTIONAL_MOC_SRC functional_dlg.h functional_objs.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tfunctional LINKS ${COMMON_LIBRARIES} FILES functional_main.cpp functional_objs.cpp ${TEST_FUNCTIONAL_MOC_SRC})
add_pv_compile_test(Tinter-threads LINKS ${COMMON_LIBRARIES} FILES inter-threads.cpp)

QT5_WRAP_CPP(TEST_DIT_MOC_SRC data-in-thread_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tdata-in-thread  LINKS ${COMMON_LIBRARIES} FILES data-in-thread.cpp ${TEST_DIT_MOC_SRC})

QT5_WRAP_CPP(TEST_MODAL_MOC_SRC modal_dlg.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tmodal LINKS ${COMMON_LIBRARIES} FILES modal.cpp ${TEST_MODAL_MOC_SRC})
add_pv_compile_test(Tmultithreads LINKS ${COMMON_LIBRARIES} FILES multithreads.cpp)
add_pv_compile_test(Thive_massive_mem LINKS ${COMMON_LIBRARIES} FILES massive-mem.cpp massive_common.cpp)
add_test(Thive_massive_mem Thive_massive_mem 1000)

###############################################################################
# Massive register checking 
###############################################################################

# object registration
add_pv_compile_test(Thive_massive_register LINKS ${COMMON_LIBRARIES} FILES massive-register.cpp massive_common.cpp)
add_test(Thive_massive_register Thive_massive_register 1000 0 0 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister0 Thive_massive_register 10000 0 0 0 0 0)
    add_test(TestHiveRegister1 Thive_massive_register 100000 0 0 0 0 0)
    add_test(TestHiveRegister2 Thive_massive_register 1000000 0 0 0 0 0)
endif()

# property registration
# varying property number
add_test(TestHiveRegister3 Thive_massive_register 1 1000 0 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister4 Thive_massive_register 1 10000 0 0 0 0)
    add_test(TestHiveRegister5 Thive_massive_register 1 100000 0 0 0 0)
    add_test(TestHiveRegister6 Thive_massive_register 1 1000000 0 0 0 0)
endif()

# varying objects/properties ratio
add_test(TestHiveRegister7 Thive_massive_register 10 1000 0 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister8 Thive_massive_register 10 100000 0 0 0 0)
    add_test(TestHiveRegister9 Thive_massive_register 100 10000 0 0 0 0)
    add_test(TestHiveRegister10 Thive_massive_register 1000 1000 0 0 0 0)
    add_test(TestHiveRegister11 Thive_massive_register 10000 100 0 0 0 0)
    add_test(TestHiveRegister12 Thive_massive_register 100000 10 0 0 0 0)
endif()

# object actors registration
# varying actors number
add_test(TestHiveRegister13 Thive_massive_register 1 0 1000 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister14 Thive_massive_register 1 0 10000 0 0 0)
    add_test(TestHiveRegister15 Thive_massive_register 1 0 100000 0 0 0)
    add_test(TestHiveRegister16 Thive_massive_register 1 0 1000000 0 0 0)
endif()

# varying objects/actors ratio
add_test(TestHiveRegister17 Thive_massive_register 10 0 1000 0 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister18 Thive_massive_register 10 0 100000 0 0 0)
    add_test(TestHiveRegister19 Thive_massive_register 100 0 10000 0 0 0)
    add_test(TestHiveRegister20 Thive_massive_register 1000 0 1000 0 0 0)
    add_test(TestHiveRegister21 Thive_massive_register 10000 0 100 0 0 0)
    add_test(TestHiveRegister22 Thive_massive_register 100000 0 10 0 0 0)
endif()

# property actors registration
# varying actors number
add_test(TestHiveRegister23 Thive_massive_register 1 1 0 1000 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister24 Thive_massive_register 1 1 0 10000 0 0)
    add_test(TestHiveRegister25 Thive_massive_register 1 1 0 100000 0 0)
    add_test(TestHiveRegister26 Thive_massive_register 1 1 0 1000000 0 0)
endif()

# varying properties/actors ratio
add_test(TestHiveRegister27 Thive_massive_register 1 10 0 1000 0 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister28 Thive_massive_register 1 10 0 100000 0 0)
    add_test(TestHiveRegister29 Thive_massive_register 1 100 0 10000 0 0)
    add_test(TestHiveRegister30 Thive_massive_register 1 1000 0 1000 0 0)
    add_test(TestHiveRegister31 Thive_massive_register 1 10000 0 100 0 0)
    add_test(TestHiveRegister32 Thive_massive_register 1 100000 0 10 0 0)
endif()
  
# object pbservers registration
# varying observers number
add_test(TestHiveRegister33 Thive_massive_register 1 0 0 0 1000 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister34 Thive_massive_register 1 0 0 0 10000 0)
    add_test(TestHiveRegister35 Thive_massive_register 1 0 0 0 100000 0)
    add_test(TestHiveRegister36 Thive_massive_register 1 0 0 0 1000000 0)
endif()
     
# varying objects/observers ratio
add_test(TestHiveRegister37 Thive_massive_register 10 0 0 0 1000 0)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister38 Thive_massive_register 10 0 0 0 100000 0)
    add_test(TestHiveRegister39 Thive_massive_register 100 0 0 0 10000 0)
    add_test(TestHiveRegister40 Thive_massive_register 1000 0 0 0 1000 0)
    add_test(TestHiveRegister41 Thive_massive_register 10000 0 0 0 100 0)
    add_test(TestHiveRegister42 Thive_massive_register 100000 0 0 0 10 0)
endif()
      
# property observers registration
# varying servers number
add_test(TestHiveRegister43 Thive_massive_register 1 1 0 0 0 1000)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister44 Thive_massive_register 1 1 0 0 0 10000)
    add_test(TestHiveRegister45 Thive_massive_register 1 1 0 0 0 100000)
    add_test(TestHiveRegister46 Thive_massive_register 1 1 0 0 0 1000000)
endif()
 
# varying properties/observers ratio
add_test(TestHiveRegister47 Thive_massive_register 1 10 0 0 0 1000)

if(DEFINED TESTS_LONG)
    add_test(TestHiveRegister48 Thive_massive_register 1 10 0 0 0 100000)
    add_test(TestHiveRegister49 Thive_massive_register 1 100 0 0 0 10000)
    add_test(TestHiveRegister50 Thive_massive_register 1 1000 0 0 0 1000)
    add_test(TestHiveRegister51 Thive_massive_register 1 10000 0 0 0 100)
    add_test(TestHiveRegister52 Thive_massive_register 1 100000 0 0 0 10)
endif()


###############################################################################
# Massive call checking 
# * Check done for call properties and call object
###############################################################################

macro(PV_MASSIVE_TEST acces)
    add_pv_compile_test(Thive_massive_call_${acces} LINKS ${COMMON_LIBRARIES} FILES massive-call-${acces}.cpp massive_common.cpp)
    add_test(Thive_massive_call_${acces} Thive_massive_call_${acces} 100 100 100)

    if(DEFINED TESTS_LONG)
	add_test(TestHiveCallObject1 Thive_massive_call_${acces} 10000000 1       0)
	add_test(TestHiveCallObject2 Thive_massive_call_${acces} 10000    1000    0)
	add_test(TestHiveCallObject3 Thive_massive_call_${acces} 10       1000000 0)

	add_test(TestHiveCallObject4 Thive_massive_call_${acces} 10000000 1       1)
	add_test(TestHiveCallObject5 Thive_massive_call_${acces} 10000    1000    1)
	add_test(TestHiveCallObject6 Thive_massive_call_${acces} 10       1000000 1)

	add_test(TestHiveCallObject7 Thive_massive_call_${acces} 1000000 1       1000)
	add_test(TestHiveCallObject8 Thive_massive_call_${acces} 1000    1000    1000)
	add_test(TestHiveCallObject9 Thive_massive_call_${acces} 1       1000000 1000)

	add_test(TestHiveCallObject10 Thive_massive_call_${acces} 1000000 1    1000000)
	add_test(TestHiveCallObject11 Thive_massive_call_${acces} 1000    1    1000000)
	add_test(TestHiveCallObject12 Thive_massive_call_${acces} 1       1000 1000000)
    endif()
endmacro()

PV_MASSIVE_TEST(obj)
PV_MASSIVE_TEST(prop)

QT5_WRAP_CPP(TEST_MQ_MOC_SRC message-queue.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tmessage-queue LINKS ${COMMON_LIBRARIES} FILES message-queue.cpp ${TEST_MQ_MOC_SRC})
add_pv_compile_test(Thive_call_helper LINKS ${COMMON_LIBRARIES} FILES call_helper.cpp)
add_test(Thive_call_helper Thive_call_helper)

QT5_WRAP_CPP(TEST_FUNC_OBSERVER_THREAD_MOC_SRC func_observer_thread.h OPTIONS ${QT5_WRAP_OPTIONS})
add_pv_compile_test(Tcore_func_observer_thread LINKS ${COMMON_LIBRARIES} FILES func_observer_thread.cpp ${TEST_FUNC_OBSERVER_THREAD_MOC_SRC})
