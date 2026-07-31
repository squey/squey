#
# @file
#

###################################################################
# Add include directories for squey's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIR})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

set(COMMON_LIBRARIES pvparallelview pvguiqt pvsquey pvkernel)

add_squey_compile_test(Tqt_listing_model LINKS ${COMMON_LIBRARIES} FILES listing_model.cpp common.cpp)

add_squey_compile_test(Tqt_layerstack LINKS ${COMMON_LIBRARIES} FILES layerstack-model.cpp common.cpp)
add_squey_compile_test(Tqt_layerstack_widget LINKS ${COMMON_LIBRARIES} FILES layerstack-widget.cpp common.cpp)
add_squey_compile_test(Tqt_vertical_tabs LINKS ${COMMON_LIBRARIES} FILES vertical_tabs.cpp common.cpp)
add_squey_compile_test(Tqt_displays_plugins LINKS ${COMMON_LIBRARIES} FILES displays_plugins.cpp common.cpp)
add_squey_test(Tqt_displays_plugins Tqt_displays_plugins -platform offscreen)
set_property(TEST SQUEY_TEST_Tqt_displays_plugins PROPERTY ENVIRONMENT "FORCE_CPU=1")

add_squey_compile_test(Tpvguiqt_sort_filter LINKS ${COMMON_LIBRARIES} FILES sort_filter.cpp)
add_squey_test(Tpvguiqt_sort_filter Tpvguiqt_sort_filter)
add_bench(Tpvguiqt_sort_filter LINKS ${COMMON_LIBRARIES} FILES sort_filter.cpp)

add_squey_compile_test(Tpvguiqt_table_view_navigation LINKS ${COMMON_LIBRARIES} FILES table_view_navigation.cpp)
add_squey_test(Tpvguiqt_table_view_navigation Tpvguiqt_table_view_navigation)

# Reports are written straight into the database, so the test links against
# Crashpad as well as the libraries above.
include_directories(AFTER SYSTEM ${CRASHPAD_INCLUDE_DIRS})
add_squey_compile_test(Tpvguiqt_crash_report_cleanup LINKS ${COMMON_LIBRARIES} ${CRASHPAD_LIBRARIES} FILES crash_report_cleanup.cpp)
add_squey_test(Tpvguiqt_crash_report_cleanup Tpvguiqt_crash_report_cleanup -platform offscreen)
# Relative for the same reason as the crash report generation test: the packaged
# testsuite runs on another machine than the one that built it.
set_tests_properties(SQUEY_TEST_Tpvguiqt_crash_report_cleanup PROPERTIES
	ENVIRONMENT "SQUEY_CONFIG_DIR=crash_report_cleanup_config")

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)