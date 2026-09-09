#
# @file
#

################################################
# Ressource declaration
################################################

set(SQUEY_ICON "resources////squey.ico")

set(MOC_HEADERS
../include/PVSingleInstanceApplication.h
../include/PVFilesTypesSelWidget.h
../include/PVMainWindow.h
../include/PVStringListChooserWidget.h
)

set(SRC_FILES
main.cpp
PVFilesTypesSelWidget.cpp
PVMainWindow.cpp
PVMainWindowMenusActions.cpp
PVMainWindowSlots.cpp
PVSaveDataTreeDialog.cpp
PVStringListChooserWidget.cpp
)

set(QRC_FILES
squey_resource.qrc
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS)
qt6_add_resources(QRC_FILES_SRCS ${QRC_FILES})

if (WIN32)
    set(EXE_OPTIONS WIN32)
    set(CMAKE_RC_COMPILE_OBJECT "<CMAKE_RC_COMPILER> -O coff -F pe-x86-64 -o <OBJECT> <SOURCE>")
    # utf8_manifest.rc sets the process code page to UTF-8, without which the
    # narrow file APIs mangle every path holding a non-ASCII character.
    set(RC_FILES_SRCS "squey_resource.rc" "${squey_SOURCE_DIR}/utf8_manifest.rc")
endif()

add_executable(squey ${EXE_OPTIONS} ${SRC_FILES} ${MOC_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS} ${RC_FILES_SRCS})
squey_speedup_compilation(squey QT)

target_include_directories(squey PRIVATE ../include)

target_link_libraries(squey PRIVATE
	pvkernel pvsquey pvparallelview pvguiqt
	Qt6::Core Qt6::Widgets Qt6::Network Qt6::Xml
	${TBB_LIBRARIES} ${ICU_LIBRARY} ${OpenMP_LIBRARIES} OpenSSL::SSL)

install(TARGETS squey DESTINATION ${CMAKE_INSTALL_BINDIR})
