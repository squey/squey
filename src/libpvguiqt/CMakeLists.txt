#
# @file
#

project(libpvguiqt)

add_definitions(-DBUILD_PVGUIQT)

include(CTest)
set(CTEST_PROJECT_NAME "PVGuiQT")

set(COPYING_DIR_NAME "COPYING")
add_definitions(-DSQUEY_COPYING_DIR="${SHARE_RESSOURCES}/${COPYING_DIR_NAME}")
install(DIRECTORY ${squey_SOURCE_DIR}/${COPYING_DIR_NAME} DESTINATION ${SHARE_RESSOURCES})

################################################
# QT6 related include and check
################################################

# For QDomElement
find_package(Qt6Xml REQUIRED)
include_directories(SYSTEM ${Qt6Xml_INCLUDE_DIRS})

# For QAction
find_package(Qt6Widgets REQUIRED)
include_directories(SYSTEM ${Qt6Widgets_INCLUDE_DIRS})

# For QGLWidget
# find_package(Qt6OpenGL REQUIRED)
# include_directories(SYSTEM ${Qt6OpenGL_INCLUDE_DIRS})

# For KSyntaxHighlighting
find_package(KF6SyntaxHighlighting REQUIRED)
get_target_property(INCLUDE_DIRS KF6::SyntaxHighlighting KF6_SYNTAXHIGHLIGHTING_INCLUDE_DIRECTORIES)
include_directories(AFTER ${KF6_SYNTAXHIGHLIGHTING_INCLUDE_DIRECTORIES})

#################################################
# Add include directories for squey's libs
#################################################

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${SQUEY_INCLUDE_DIRS})
include_directories(AFTER ${PVDISPLAYS_INCLUDE_DIRS})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIRS})
include_directories(AFTER ${PVLICENSE_INCLUDE_DIR})
include_directories(AFTER SYSTEM ${SIGCPP_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)