#
# @file
#

################################################
# Ressource declaration
################################################

set(LIBGUIQT_SRC_FILES
common.cpp
PVAxesCombinationDialog.cpp
PVAxesCombinationWidget.cpp
PVCredentialDialog.cpp
PVHiveDataTreeModel.cpp
PVLayerFilterProcessWidget.cpp
PVLayerStackDelegate.cpp
PVLayerStackModel.cpp
PVLayerStackView.cpp
PVLayerStackWidget.cpp
PVListDisplayDlg.cpp
PVAbstractListStatsDlg.cpp
PVGroupByStringsDlg.cpp
PVListingModel.cpp
PVAbstractTableModel.cpp
PVListingView.cpp
PVAbstractTableView.cpp
PVRootTreeModel.cpp
PVRootTreeView.cpp
PVTableView.cpp
PVQNraw.cpp
PVQMapped.cpp
PVQPlotted.cpp
PVWorkspacesTabWidget.cpp
PVWorkspace.cpp
PVViewDisplay.cpp
PVAboutBoxDialog.cpp
PVProjectsTabWidget.cpp
PVStartScreenWidget.cpp
PVInputTypeMenuEntries.cpp
PVImportSourceToProjectDlg.cpp
PVStatsListingWidget.cpp
PVExportSelectionDlg.cpp
PVChangelogMessage.cpp
PVPythonScriptWidget.cpp
PVPythonCodeEditor.cpp

# Displays that will be registered to libpvdisplays
PVDisplaySourceDataTree.cpp
PVDisplayViewAxesCombination.cpp
PVDisplayViewLayerStack.cpp
PVDisplayViewListing.cpp
PVDisplayViewPythonConsole.cpp
)

SET(LIBGUIQT_MOC_HEADERS
include/pvguiqt/PVAxesCombinationDialog.h
include/pvguiqt/PVAxesCombinationWidget.h
include/pvguiqt/PVLayerFilterProcessWidget.h
include/pvguiqt/PVLayerStackDelegate.h
include/pvguiqt/PVLayerStackModel.h
include/pvguiqt/PVLayerStackView.h
include/pvguiqt/PVLayerStackWidget.h
include/pvguiqt/PVListDisplayDlg.h
include/pvguiqt/PVAbstractListStatsDlg.h
include/pvguiqt/PVListingModel.h
include/pvguiqt/PVAbstractTableModel.h
include/pvguiqt/PVListingView.h
include/pvguiqt/PVAbstractTableView.h
include/pvguiqt/PVTableView.h
include/pvguiqt/PVRootTreeModel.h
include/pvguiqt/PVRootTreeView.h
include/pvguiqt/PVHiveDataTreeModel.h
include/pvguiqt/PVWorkspacesTabWidget.h
include/pvguiqt/PVWorkspace.h
include/pvguiqt/PVViewDisplay.h
include/pvguiqt/PVProjectsTabWidget.h
include/pvguiqt/PVStartScreenWidget.h
include/pvguiqt/PVImportSourceToProjectDlg.h
include/pvguiqt/PVStatsListingWidget.h
include/pvguiqt/PVExportSelectionDlg.h
include/pvguiqt/PVAboutBoxDialog.h
include/pvguiqt/PVPythonScriptWidget.h
include/pvguiqt/PVPythonCodeEditor.h
)

SET(UI_FILES
PVAxesCombinationWidget.ui
PVListDisplayDlg.ui
)

SET(QRC_FILES
../../gui-qt/src/inendi-inspector_resource.qrc
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${LIBGUIQT_SRC_FILES})
set(HEADER_FILES ${LIBGUIQT_HEADER_FILES})
set(MOC_HEADERS ${LIBGUIQT_MOC_HEADERS})
QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})
QT6_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})
PV_QT6_WRAP_UI(UI_SRC pvguiqt ${UI_FILES})

################################################
# Declare the new library and linkage options
################################################

add_library(pvguiqt SHARED ${SRC_FILES} ${MOC_SRC} ${QRC_FILES_SRCS} ${UI_FILES})

set_target_properties(pvguiqt PROPERTIES SOVERSION 1)

set(QT6_LIBS Qt6Core Qt6Xml Qt6Gui Qt6Widgets Qt6OpenGL Qt63DCore Qt63DRender Qt63DInput Qt63DExtras)

set(PVGUIQT_LINK_LIBRARIES pvdisplays inendi pvkernel ${PVHWLOC_LIBRARIES} ${PVLOGGER_LIBRARIES} ${OPENGL_LIBRARIES}
	${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${QT6_LIBS} /app/lib/libKF5SyntaxHighlighting.so)

target_link_libraries(pvguiqt ${PVGUIQT_LINK_LIBRARIES})

INSTALL(TARGETS pvguiqt LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
                        
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Core_EXECUTABLE_COMPILE_FLAGS}")

################################################
# Crash reporter
################################################
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

compile_and_install_tool(inendi-crashreport LINKS ${LINK_LIBRARIES} ${QT6_LIBS} pvkernel curl FILES crash_report.cpp)


