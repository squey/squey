#
# @file
#

include_directories(AFTER SYSTEM ${ICU_INCLUDE})

################################################
# Ressource declaration
################################################

set(SQUEY_ICON "resources////squey.ico")

SET(MOC_HEADERS
../include/PVSingleInstanceApplication.h
../include/PVFilesTypesSelWidget.h
../include/PVMainWindow.h
../include/PVStringListChooserWidget.h
)

SET(SRC_FILES
main.cpp
PVFilesTypesSelWidget.cpp
PVMainWindow.cpp
PVMainWindowMenusActions.cpp
PVMainWindowSlots.cpp
PVSaveDataTreeDialog.cpp
PVStringListChooserWidget.cpp
)

SET(QRC_FILES
squey_resource.qrc
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})
QT6_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})

if (WIN32)
    set(EXE_OPTIONS WIN32)
    set(CMAKE_RC_COMPILE_OBJECT "<CMAKE_RC_COMPILER> -O coff -F pe-x86-64 -o <OBJECT> <SOURCE>")
    set(RC_FILES_SRCS "squey_resource.rc")
endif()

add_executable(squey ${EXE_OPTIONS} ${SRC_FILES} ${MOC_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS} ${RC_FILES_SRCS})

set_property(TARGET squey APPEND PROPERTY LINK_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Widgets_EXECUTABLE_COMPILE_FLAGS}")

set(PV_QT_LIBS Qt6Core Qt6Widgets Qt6Network Qt6Xml Qt6Widgets)

target_link_libraries(squey ssl ${QT_LIBRARIES} ${PVPARALLELVIEW_LIBRARIES}
    pvkernel pvsquey pvparallelview pvguiqt ${TBB_LIBRARIES} ${ICU_LIBRARY}
    ${OpenMP_LIBRARIES} ${PV_QT_LIBS})

INSTALL(TARGETS squey DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
