#
# @file
#

project(libpvkernel)
include(CTest)
set(CTEST_PROJECT_NAME "PVKernel")
add_compile_definitions(SINGLETON_DLL)

################################################
# QT6 related configuration
################################################
# The components are looked up once in CMakeRequiredLibraries.txt; what is left
# here is the choice this directory makes about OpenGL.

# FIXME : This is a hack waiting for Qt 5.4 and QOpenGLWidgets
set(QT_NO_OPENGL 1)
if(QT_NO_OPENGL)
	add_compile_definitions(QT_NO_OPENGL)
else()
	# For QGLWidget
	find_package(Qt6 REQUIRED COMPONENTS OpenGL)
endif()

# QUIET: ArrowConfig.cmake first tries config-mode find_package calls for lz4
# and zstd, which warn when no CMake package exists for them (the freedesktop
# SDK ships none) before the pkg-config fallback finds them anyway.
find_package(Arrow CONFIG REQUIRED QUIET)
find_package(Parquet CONFIG REQUIRED QUIET)
message(STATUS "Found Arrow ${Arrow_VERSION} and Parquet ${Parquet_VERSION}")
find_package(Thrift CONFIG REQUIRED)
find_package(re2 REQUIRED)
set(PARQUET_LIBS Arrow::arrow_shared Parquet::parquet_shared thrift re2::re2 snappy)

##################################################################
# set installation path for plugins
##################################################################
add_compile_definitions(PVKERNEL_PLUGIN_PATH="${PVKERNEL_PLUGIN}")

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)
add_subdirectory(tools)

set(PVKERNEL_PLUGINS_LIST ${PLUGINS_LIST})

add_subdirectory(tests)

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)
