#
# @file
#

###################################################################
# Run Qt integration tests
###################################################################

find_package(Qt6Test REQUIRED)
include_directories(SYSTEM ${Qt6Test_INCLUDE_DIRS})

include_directories(AFTER ${PVKERNEL_INCLUDE_DIR})
include_directories(AFTER ${PVGUIQT_INCLUDE_DIR})
include_directories(AFTER ${PVPARALLELVIEW_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

set(COMMON_LIBRARIES pvparallelview pvguiqt pvsquey pvkernel Qt6::Test)

SET(MOC_HEADERS
../include/PVFilesTypesSelWidget.h
../include/PVMainWindow.h
../include/PVStringListChooserWidget.h
import_export.h
)

SET(SRC_FILES
../src/PVFilesTypesSelWidget.cpp
../src/PVMainWindow.cpp
../src/PVMainWindowMenusActions.cpp
../src/PVMainWindowSlots.cpp
../src/PVSaveDataTreeDialog.cpp
../src/PVStringListChooserWidget.cpp
)

QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})

#add_squey_compile_test(Tpvguiqt_import_export LINKS ${COMMON_LIBRARIES} FILES ${MOC_SRC} ${SRC_FILES} import_export.cpp)
#add_squey_test(Tpvguiqt_import_export Tpvguiqt_import_export -platform offscreen)

file(RELATIVE_PATH REL_TEST_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake DESTINATION "tests/${REL_TEST_DIR}" COMPONENT tests)