#
# @file
#

include_directories(AFTER ../include/pvguiqt/format-builder)

################################################
# Ressource declaration
################################################

set(LIBGUIQT_SRC_FILES
common.cpp
PVAxesCombinationDialog.cpp
PVAxesCombinationWidget.cpp
PVCredentialDialog.cpp
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
PVTableView.cpp
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
PVDockWidgetTitleBar.cpp
PVStatusBar.cpp
PVImportWorkflowTabBar.cpp
PVErrorsAndWarnings.cpp

# FormatBuilder
format-builder/PVFormatBuilderWidget.cpp
format-builder/PVNrawListingModel.cpp
format-builder/PVNrawListingWidget.cpp
format-builder/PVXmlDomModel.cpp
format-builder/PVXmlParamColorDialog.cpp
format-builder/PVXmlParamComboBox.cpp
format-builder/PVXmlParamList.cpp
format-builder/PVXmlParamTextEdit.cpp
format-builder/PVXmlParamWidget.cpp
format-builder/PVXmlParamWidgetBoardAxis.cpp
format-builder/PVXmlParamWidgetBoardFilter.cpp
format-builder/PVXmlParamWidgetBoardSplitterRegEx.cpp
format-builder/PVXmlParamWidgetEditorBox.cpp
format-builder/PVXmlRegValidatorHighLight.cpp
format-builder/PVXmlTreeItemDelegate.cpp
format-builder/PVXmlTreeView.cpp
format-builder/PVXmlTimeValidatorHighLight.cpp
format-builder/PVOptionsWidget.cpp

# Displays that will be registered to libpvdisplays
PVDisplayViewCorrelation.cpp
PVDisplayViewAxesCombination.cpp
PVDisplayViewMappingPlotting.cpp
PVDisplayViewDistinctValues.cpp
PVDisplayViewGroupBy.cpp
PVDisplayViewLayerStack.cpp
PVDisplayViewListing.cpp
PVDisplayViewPythonConsole.cpp
PVProgressBoxPython.cpp
)

SET(LIBGUIQT_MOC_HEADERS
../include/pvguiqt/PVAxesCombinationDialog.h
../include/pvguiqt/PVAxesCombinationWidget.h
../include/pvguiqt/PVLayerFilterProcessWidget.h
../include/pvguiqt/PVLayerStackDelegate.h
../include/pvguiqt/PVLayerStackModel.h
../include/pvguiqt/PVLayerStackView.h
../include/pvguiqt/PVLayerStackWidget.h
../include/pvguiqt/PVListDisplayDlg.h
../include/pvguiqt/PVAbstractListStatsDlg.h
../include/pvguiqt/PVListingModel.h
../include/pvguiqt/PVAbstractTableModel.h
../include/pvguiqt/PVListingView.h
../include/pvguiqt/PVAbstractTableView.h
../include/pvguiqt/PVTableView.h
../include/pvguiqt/PVWorkspacesTabWidget.h
../include/pvguiqt/PVWorkspace.h
../include/pvguiqt/PVViewDisplay.h
../include/pvguiqt/PVProjectsTabWidget.h
../include/pvguiqt/PVStartScreenWidget.h
../include/pvguiqt/PVImportSourceToProjectDlg.h
../include/pvguiqt/PVStatsListingWidget.h
../include/pvguiqt/PVExportSelectionDlg.h
../include/pvguiqt/PVAboutBoxDialog.h
../include/pvguiqt/PVPythonScriptWidget.h
../include/pvguiqt/PVPythonCodeEditor.h
../include/pvguiqt/PVProgressBoxPython.h
../include/pvguiqt/PVDockWidgetTitleBar.h
../include/pvguiqt/PVStatusBar.h
../include/pvguiqt/PVErrorsAndWarnings.h

# FormatBuilder
../include/pvguiqt/format-builder/PVFormatBuilderWidget.h
../include/pvguiqt/format-builder/PVNrawListingWidget.h
../include/pvguiqt/format-builder/PVXmlTreeView.h
../include/pvguiqt/format-builder/PVXmlParamColorDialog.h
../include/pvguiqt/format-builder/PVXmlParamComboBox.h
../include/pvguiqt/format-builder/PVXmlParamList.h
../include/pvguiqt/format-builder/PVXmlParamTextEdit.h
../include/pvguiqt/format-builder/PVXmlParamWidget.h
../include/pvguiqt/format-builder/PVXmlParamWidgetBoardAxis.h
../include/pvguiqt/format-builder/PVXmlParamWidgetBoardFilter.h
../include/pvguiqt/format-builder/PVXmlParamWidgetBoardSplitterRegEx.h
../include/pvguiqt/format-builder/PVXmlRegValidatorHighLight.h
../include/pvguiqt/format-builder/PVXmlTimeValidatorHighLight.h
../include/pvguiqt/format-builder/PVOptionsWidget.h
)

SET(UI_FILES
PVAxesCombinationWidget.ui
PVListDisplayDlg.ui
)

SET(QRC_FILES
../../gui-qt/src/squey_resource.qrc
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

set(QT6_LIBS Qt6Core Qt6Xml Qt6Gui Qt6Widgets Qt6OpenGL)

set(PVGUIQT_LINK_LIBRARIES pvdisplays pvsquey pvkernel ${PVHWLOC_LIBRARIES} ${PVLOGGER_LIBRARIES} ${OPENGL_LIBRARIES}
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

compile_and_install_tool(squey-crashreport LINKS ${LINK_LIBRARIES} ${QT6_LIBS} pvkernel curl FILES crash_report.cpp)


