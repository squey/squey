#
# @file
#

# The App Store version of macOS loads no kernel but those it ships, which
# macos_publish.sh builds with this in each bundle, then removes it from them.
# Built everywhere, so that the build of the other platforms checks it too.
add_executable(squey-prepare-kernels prepare_kernels.cpp)
target_link_libraries(squey-prepare-kernels PRIVATE pvparallelview)
if (APPLE)
	install(TARGETS squey-prepare-kernels DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
