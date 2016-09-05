#
# @file
#
# @copyright (C) ESI Group INENDI 2016

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

function(compile_and_install_tool tool_exe)
	# Define all kinds of arguments
	set(options)
	set(oneValueArgs)
	set(multiValueArgs FILES LINKS)
	
	# Call cmake_parse_arguments to extract information in variables
	cmake_parse_arguments(_WRAP_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	set(args ${_WRAP_TEST_UNPARSED_ARGUMENTS})
	set(files ${_WRAP_TEST_FILES})
	set(links ${_WRAP_TEST_LINKS})

	add_executable(${tool_exe} ${files})
	target_link_libraries(${tool_exe} ${links})
	INSTALL(TARGETS ${tool_exe} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endfunction()

add_subdirectory(rush)
