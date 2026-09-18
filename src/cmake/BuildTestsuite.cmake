# Builds what the test suite runs: the default target, which holds the plugins the
# tests load, and the test executables themselves. ctest runs it before the tests.
#
# Usage: cmake -DBUILD_DIR=<build tree> -P BuildTestsuite.cmake
#
# An installed copy of the tests has no build tree, and nothing to build.

if (EXISTS "${BUILD_DIR}/CMakeCache.txt")
	execute_process(
		COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target all squey_testsuite
		RESULT_VARIABLE result)
	if (NOT result EQUAL 0)
		message(FATAL_ERROR "The test suite does not build")
	endif()
endif()
