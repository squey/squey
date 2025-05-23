#
# @file
#

project(libpvkernel)
include(CTest)
set(CTEST_PROJECT_NAME "PVKernel")
add_definitions(-DSINGLETON_DLL)

################################################
# QT6 related include and check
################################################

# For QDomElement
find_package(Qt6Xml REQUIRED)
include_directories(SYSTEM ${Qt6Xml_INCLUDE_DIRS})

# For colors
find_package(Qt6Gui REQUIRED)
include_directories(SYSTEM ${Qt6Gui_INCLUDE_DIRS})

# For Dialog
find_package(Qt6Widgets REQUIRED)
include_directories(SYSTEM ${Qt6Widgets_INCLUDE_DIRS})

# For Qt6Core5Compat
find_package(Qt6Core5Compat REQUIRED)
include_directories(SYSTEM ${Qt6Core5Compat_INCLUDE_DIRS})

# FIXME : This is a hack waiting for Qt 5.4 and QOpenGLWidgets
set(QT_NO_OPENGL 1)
if(${QT_NO_OPENGL})
	add_definitions("-DQT_NO_OPENGL")
else(${QT_NO_OPENGL})
# For QGLWidget
find_package(Qt6OpenGL REQUIRED)
include_directories(SYSTEM ${Qt6OpenGL_INCLUDE_DIRS})
endif(${QT_NO_OPENGL})

# For QtWebKit or QtWebEngine
if(${QT_WEBKIT})
	find_package(Qt6WebKitWidgets REQUIRED)
	include_directories(SYSTEM ${Qt6WebKitWidgets_INCLUDE_DIRS})
else(${QT_WEBKIT})
	# find_package(Qt6WebEngineWidgets REQUIRED)
	# include_directories(SYSTEM ${Qt6WebEngineWidgets_INCLUDE_DIRS})
	# find_package(Qt6WebEngineCore REQUIRED)
	# include_directories(SYSTEM ${Qt6WebEngineCore_INCLUDE_DIRS})
endif(${QT_WEBKIT})

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
find_package(Thrift CONFIG REQUIRED)
find_package(re2 REQUIRED)
set(PARQUET_LIBS Arrow::arrow_shared Parquet::parquet_shared thrift re2::re2 snappy)

find_package(Qt6DBus REQUIRED)
include_directories(SYSTEM ${Qt6DBus_INCLUDE_DIRS})


##################################################################
# set installation path for plugins
##################################################################
add_definitions(-DPVKERNEL_PLUGIN_PATH="${PVKERNEL_PLUGIN}")

###################################################################
# Add include directories for squey's libs and external projects
###################################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER SYSTEM ${ICU_INCLUDE})
include_directories(AFTER SYSTEM ${ARCHIVE_INCLUDE_DIRS})
include_directories(AFTER SYSTEM ${BREAKPAD_INCLUDE_DIRS})
include_directories(AFTER SYSTEM ${Boost_INCLUDE_DIR})
include_directories(AFTER SYSTEM ${HWLOC_INCLUDE_DIRS})
include_directories(AFTER SYSTEM ${SIGCPP_INCLUDE_DIRS})
include_directories(AFTER "src/third_party/uchardetect")

###############################################################################
# Configure subdirectories
###############################################################################

add_subdirectory(src)
add_subdirectory(plugins)
add_subdirectory(tools)

set(PVKERNEL_PLUGINS_LIST ${PLUGINS_LIST})

add_subdirectory(tests)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)