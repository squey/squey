#
# @file
#

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
add_squey_compile_test(Tcore_file_helper LINKS ${LINK_LIBRARIES} FILES file_helper.cpp)
add_squey_test(Tcore_file_helper Tcore_file_helper)
endif()

add_squey_compile_test(Tcore_directory_remove LINKS ${LINK_LIBRARIES} FILES directory_remove.cpp)
add_squey_test(Tcore_directory_remove Tcore_directory_remove)

add_squey_compile_test(Tcore_intrinsics LINKS ${LINK_LIBRARIES} FILES intrinsics.cpp)
add_squey_test(Tcore_intrinsics Tcore_intrinsics)

add_squey_compile_test(Tcore_visit_bits LINKS ${LINK_LIBRARIES} ${OpenMP_LIBRARIES} FILES visit_bits.cpp)
add_squey_test(Tcore_visit_bits Tcore_visit_bits)

add_squey_compile_test(Tcore_bit_field_visit LINKS ${LINK_LIBRARIES} ${OpenMP_LIBRARIES} FILES bit_field_visit.cpp)
add_squey_test(Tcore_bit_field_visit Tcore_bit_field_visit)

add_squey_compile_test(Tcore_argument_serialize LINKS ${LINK_LIBRARIES} FILES argument_serialize.cpp)
add_squey_test(Tcore_argument_serialize Tcore_argument_serialize)

add_squey_compile_test(Tcore_network LINKS ${LINK_LIBRARIES} FILES network.cpp)
add_squey_test(Tcore_network Tcore_network)

add_squey_compile_test(Tcore_timeformat LINKS ${LINK_LIBRARIES} ${ICU_LIBRARY} FILES time-format.cpp)
add_squey_test(Tcore_timeformat Tcore_timeformat ${TESTS_FILES_DIR}/pvkernel/core/time_format)

add_squey_compile_test(Tcore_crash_reporter LINKS ${LINK_LIBRARIES} curl FILES crash_reporter.cpp)
add_squey_test(Tcore_crash_reporter Tcore_crash_reporter)

add_squey_compile_test(Tcore_crash_report_generation LINKS ${LINK_LIBRARIES} FILES crash_report_generation.cpp)
add_squey_test(Tcore_crash_report_generation Tcore_crash_report_generation)
# The test re-executes itself to crash, and the child looks the handler up next
# to its own executable, as squey does.
add_custom_command(TARGET SQUEY_TEST_Tcore_crash_report_generation POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:crashpad::crashpad_handler>
		$<TARGET_FILE_DIR:SQUEY_TEST_Tcore_crash_report_generation>)
file(RELATIVE_PATH CRASH_TEST_REL_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(PROGRAMS $<TARGET_FILE:crashpad::crashpad_handler>
	DESTINATION "tests/${CRASH_TEST_REL_DIR}" COMPONENT tests OPTIONAL)
# A configuration directory of its own, so that the test neither reads nor
# writes the crash reports of whoever runs it. The path is relative on purpose:
# PVConfig reads the variable when the library is loaded, so it has to come from
# here, and an absolute build path would not exist on the machine that runs the
# packaged testsuite of Windows and macOS.
# 77 is returned where ptrace is forbidden, which keeps Crashpad from capturing
# anything: the test reports a skip rather than a failure there.
set_tests_properties(SQUEY_TEST_Tcore_crash_report_generation PROPERTIES SKIP_RETURN_CODE 77)
set_property(TEST SQUEY_TEST_Tcore_crash_report_generation APPEND PROPERTY
	ENVIRONMENT "SQUEY_CONFIG_DIR=crash_report_generation_config")

add_squey_compile_test(Tcore_streaming_compressor LINKS ${LINK_LIBRARIES} FILES streaming_compressor.cpp)
add_squey_test(Tcore_streaming_compressor Tcore_streaming_compressor)

# An OpenCL driver standing in for one that misbehaves, for the tests of the
# OpenCL start-up. They have the loader of the Linux runtime, ocl-icd, load it:
# the Windows one ignores the environment in the elevated process the CI runs
# the tests in, and macOS has no loader.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	add_library(squey_test_opencl_icd MODULE EXCLUDE_FROM_ALL opencl_test_icd.cpp)
	# The compile definitions of Squey::OpenCL, not the target itself: a driver
	# has no business linking against the loader that loads it.
	target_compile_definitions(squey_test_opencl_icd PRIVATE
		$<TARGET_PROPERTY:Squey::OpenCL,INTERFACE_COMPILE_DEFINITIONS>)

	add_squey_compile_test(Tcore_opencl_device_type LINKS ${LINK_LIBRARIES} FILES opencl_device_type.cpp)
	target_compile_definitions(SQUEY_TEST_Tcore_opencl_device_type PRIVATE
		SQUEY_TEST_OPENCL_ICD=\"$<TARGET_FILE:squey_test_opencl_icd>\")
	add_dependencies(SQUEY_TEST_Tcore_opencl_device_type squey_test_opencl_icd)
	add_squey_test(Tcore_opencl_device_type Tcore_opencl_device_type)

	# The same driver standing for a hybrid laptop instead, its GPU built into
	# the processor listed before its dedicated one.
	add_library(squey_test_opencl_icd_hybrid MODULE EXCLUDE_FROM_ALL opencl_test_icd.cpp)
	target_compile_definitions(squey_test_opencl_icd_hybrid PRIVATE SQUEY_TEST_ICD_HYBRID
		$<TARGET_PROPERTY:Squey::OpenCL,INTERFACE_COMPILE_DEFINITIONS>)

	add_squey_compile_test(Tcore_opencl_dedicated_gpu LINKS ${LINK_LIBRARIES} FILES opencl_dedicated_gpu.cpp)
	target_compile_definitions(SQUEY_TEST_Tcore_opencl_dedicated_gpu PRIVATE
		SQUEY_TEST_OPENCL_ICD=\"$<TARGET_FILE:squey_test_opencl_icd_hybrid>\")
	add_dependencies(SQUEY_TEST_Tcore_opencl_dedicated_gpu squey_test_opencl_icd_hybrid)
	add_squey_test(Tcore_opencl_dedicated_gpu Tcore_opencl_dedicated_gpu)
endif()

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)