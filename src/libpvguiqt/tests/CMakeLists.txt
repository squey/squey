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
add_squey_compile_test(Tqt_displays_plugins LINKS ${COMMON_LIBRARIES} FILES displays_plugins.cpp common.cpp)
add_squey_test(Tqt_displays_plugins Tqt_displays_plugins -platform offscreen)
set_property(TEST SQUEY_TEST_Tqt_displays_plugins APPEND PROPERTY ENVIRONMENT "FORCE_CPU=1")

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