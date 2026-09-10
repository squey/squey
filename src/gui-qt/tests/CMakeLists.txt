#
# @file
#

###################################################################
# Run Qt integration tests

set(COMMON_LIBRARIES pvparallelview pvguiqt pvsquey pvkernel Qt6::Test)

set(MOC_HEADERS
../include/PVFilesTypesSelWidget.h
../include/PVMainWindow.h
../include/PVStringListChooserWidget.h
import_export.h
)

set(SRC_FILES
../src/PVFilesTypesSelWidget.cpp
../src/PVMainWindow.cpp
../src/PVMainWindowMenusActions.cpp
../src/PVMainWindowSlots.cpp
../src/PVSaveDataTreeDialog.cpp
../src/PVStringListChooserWidget.cpp
)

qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS)

add_squey_compile_test(Tpvguiqt_import_export LINKS ${COMMON_LIBRARIES} FILES ${MOC_SRC} ${SRC_FILES} import_export.cpp)
# The test compiles the sources of the application, which include its headers
# by name.
target_include_directories(SQUEY_TEST_Tpvguiqt_import_export
	PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}/../src")
add_squey_test(Tpvguiqt_import_export Tpvguiqt_import_export -platform offscreen)
set_property(TEST SQUEY_TEST_Tpvguiqt_import_export PROPERTY ENVIRONMENT "FORCE_CPU=1")

###################################################################
# The handover of files between instances, which "open with" rests on

qt6_wrap_cpp(SINGLE_INSTANCE_MOC_SRC ../include/PVSingleInstanceApplication.h single_instance.h OPTIONS)

add_squey_compile_test(Tpvguiqt_single_instance LINKS pvkernel Qt6::Test Qt6::Widgets Qt6::Network FILES ${SINGLE_INSTANCE_MOC_SRC} single_instance.cpp)
target_include_directories(SQUEY_TEST_Tpvguiqt_single_instance
	PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}")
add_squey_test(Tpvguiqt_single_instance Tpvguiqt_single_instance -platform offscreen)

file(RELATIVE_PATH REL_TEST_DIR ${squey_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)
