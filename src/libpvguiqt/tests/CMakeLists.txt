#
# @file
#

###################################################################
# Add include directories for squey's libs and external projects
###################################################################

set(COMMON_LIBRARIES pvparallelview pvguiqt pvsquey pvkernel)

add_squey_compile_test(Tqt_listing_model LINKS ${COMMON_LIBRARIES} FILES listing_model.cpp common.cpp)

add_squey_compile_test(Tqt_layerstack LINKS ${COMMON_LIBRARIES} FILES layerstack-model.cpp common.cpp)
add_squey_compile_test(Tqt_layerstack_widget LINKS ${COMMON_LIBRARIES} FILES layerstack-widget.cpp common.cpp)
add_squey_compile_test(Tqt_vertical_tabs LINKS ${COMMON_LIBRARIES} FILES vertical_tabs.cpp common.cpp)
# dbghelp backs the crash backtrace this test prints on Windows, where ctest
# reports nothing beyond "SEGFAULT".
if(WIN32)
	set(DISPLAYS_PLUGINS_LIBRARIES ${COMMON_LIBRARIES} dbghelp)
else()
	set(DISPLAYS_PLUGINS_LIBRARIES ${COMMON_LIBRARIES})
endif()
add_squey_compile_test(Tqt_displays_plugins LINKS ${DISPLAYS_PLUGINS_LIBRARIES} FILES displays_plugins.cpp common.cpp)
add_squey_test(Tqt_displays_plugins Tqt_displays_plugins -platform offscreen)
set_property(TEST SQUEY_TEST_Tqt_displays_plugins APPEND PROPERTY ENVIRONMENT "FORCE_CPU=1")

add_squey_compile_test(Tqt_sql_console_result LINKS ${COMMON_LIBRARIES} FILES sql_console_result.cpp common.cpp)
add_squey_test(Tqt_sql_console_result Tqt_sql_console_result -platform offscreen)

# Simulated mouse events: the click is the feature, so it is what the test has
# to send.
find_package(Qt6Test REQUIRED)
include_directories(SYSTEM ${Qt6Test_INCLUDE_DIRS})
# Qt6::Widgets explicitly: QTest's widget overloads are compiled in by
# QT_WIDGETS_LIB, which only linking that target defines.
add_squey_compile_test(Tqt_sql_completion LINKS ${COMMON_LIBRARIES} Qt6::Test Qt6::Widgets FILES sql_completion.cpp)
add_squey_test(Tqt_sql_completion Tqt_sql_completion -platform offscreen)

add_squey_compile_test(Tpvguiqt_sort_filter LINKS ${COMMON_LIBRARIES} FILES sort_filter.cpp)
add_squey_test(Tpvguiqt_sort_filter Tpvguiqt_sort_filter)
add_bench(Tpvguiqt_sort_filter LINKS ${COMMON_LIBRARIES} FILES sort_filter.cpp)

# Linux only, the OpenCL driver it loads being built there only (see the tests of libpvkernel).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	add_squey_compile_test(Tpvguiqt_about_box_opencl LINKS ${COMMON_LIBRARIES} FILES about_box_opencl.cpp)
	target_compile_definitions(SQUEY_TEST_Tpvguiqt_about_box_opencl PRIVATE
		SQUEY_TEST_OPENCL_ICD=\"$<TARGET_FILE:squey_test_opencl_icd>\")
	add_dependencies(SQUEY_TEST_Tpvguiqt_about_box_opencl squey_test_opencl_icd)
	add_squey_test(Tpvguiqt_about_box_opencl Tpvguiqt_about_box_opencl -platform offscreen)
endif()

add_squey_compile_test(Tpvguiqt_table_view_navigation LINKS ${COMMON_LIBRARIES} FILES table_view_navigation.cpp)
add_squey_test(Tpvguiqt_table_view_navigation Tpvguiqt_table_view_navigation)

add_squey_compile_test(Tpvguiqt_listing_model_outlived LINKS ${COMMON_LIBRARIES} FILES listing_model_outlived.cpp common.cpp)
add_squey_test(Tpvguiqt_listing_model_outlived Tpvguiqt_listing_model_outlived -platform offscreen)

# Reports are written straight into the database, so the test links against
# Crashpad as well as the libraries above.
include_directories(AFTER SYSTEM ${CRASHPAD_INCLUDE_DIRS})
add_squey_compile_test(Tpvguiqt_crash_report_cleanup LINKS ${COMMON_LIBRARIES} ${CRASHPAD_LIBRARIES} FILES crash_report_cleanup.cpp)
add_squey_test(Tpvguiqt_crash_report_cleanup Tpvguiqt_crash_report_cleanup -platform offscreen)
# Relative for the same reason as the crash report generation test: the packaged
# testsuite runs on another machine than the one that built it.
set_property(TEST SQUEY_TEST_Tpvguiqt_crash_report_cleanup APPEND PROPERTY
	ENVIRONMENT "SQUEY_CONFIG_DIR=crash_report_cleanup_config")

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)