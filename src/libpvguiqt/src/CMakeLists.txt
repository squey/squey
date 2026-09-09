#
# @file
#

#set(CMAKE_CXX_FLAGS "-Wl,--version-script=${squey_SOURCE_DIR}/libpvguiqt/export.def ${CMAKE_CXX_FLAGS}")

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
PVDisplayViewMappingScaling.cpp
PVDisplayViewDistinctValues.cpp
PVDisplayViewGroupBy.cpp
PVDisplayViewLayerStack.cpp
PVDisplayViewListing.cpp
PVDisplayViewFilters.cpp
)

set(LIBGUIQT_MOC_HEADERS
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

if (PYTHON_SUPPORT)
	set(LIBGUIQT_SRC_FILES ${LIBGUIQT_SRC_FILES}
	PVProgressBoxPython.cpp
	PVDisplayViewPythonConsole.cpp
	)
	set(LIBGUIQT_MOC_HEADERS ${LIBGUIQT_MOC_HEADERS}
	../include/pvguiqt/PVProgressBoxPython.h
	)
endif()

set(UI_FILES
PVAxesCombinationWidget.ui
PVListDisplayDlg.ui
)

set(QRC_FILES
../../gui-qt/src/squey_resource.qrc
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${LIBGUIQT_SRC_FILES})
set(HEADER_FILES ${LIBGUIQT_HEADER_FILES})
set(MOC_HEADERS ${LIBGUIQT_MOC_HEADERS})
qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS)
qt6_add_resources(QRC_FILES_SRCS ${QRC_FILES})
pv_qt6_wrap_ui(UI_SRC pvguiqt ${UI_FILES})

################################################
# Declare the new library and linkage options
################################################

add_library(pvguiqt SHARED ${SRC_FILES} ${MOC_SRC} ${QRC_FILES_SRCS} ${UI_FILES})
target_compile_definitions(pvguiqt PRIVATE PVGUIQT_LIBRARY)
set_target_properties(pvguiqt PROPERTIES SOVERSION 1)
squey_speedup_compilation(pvguiqt QT)

target_include_directories(pvguiqt PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/pvguiqt/format-builder>)

target_link_libraries(pvguiqt
	PUBLIC
		pvkernel pvsquey pvdisplays pvparallelview
		Qt6::Core Qt6::Xml Qt6::Gui Qt6::Widgets Qt6::Core5Compat
		KF6::SyntaxHighlighting
	PRIVATE
		${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS pvguiqt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
				RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

################################################
# Crash reporter
################################################
# On macOS the crash reporter links against the libcurl of the system rather than
# the one of the prefix. Ours is built against OpenSSL with no CA bundle, and
# macOS provides no store for it, so verifying the certificate of the crash
# server always fails. The libcurl of the system goes through the keychain, which
# macOS keeps up to date -- as Windows does through CURLSSLOPT_NATIVE_CA and
# Linux through the certificates of the flatpak runtime. The reporter is a
# program of its own and loads neither libsquey nor the elasticsearch plugins, so
# no second libcurl ends up in the same process.
if (APPLE)
	# A native build names the SDK through CMAKE_OSX_SYSROOT, while the osxcross
	# toolchain leaves that empty and exposes OSXCROSS_SDK instead.
	if (CMAKE_OSX_SYSROOT)
		set(MACOS_SDK_DIR "${CMAKE_OSX_SYSROOT}")
	else()
		set(MACOS_SDK_DIR "$ENV{OSXCROSS_SDK}")
	endif()
	set(CRASH_REPORTER_CURL "${MACOS_SDK_DIR}/usr/lib/libcurl.tbd")
	if (NOT EXISTS "${CRASH_REPORTER_CURL}")
		message(FATAL_ERROR "libcurl of the macOS SDK not found: ${CRASH_REPORTER_CURL}")
	endif()
else()
	set(CRASH_REPORTER_CURL CURL::libcurl)
endif()

compile_and_install_tool(squey-crashreport
	LINKS pvkernel Qt6::Core Qt6::Widgets ${CRASH_REPORTER_CURL}
	FILES crash_report.cpp)
# PVCrashReporterDialog.h has no implementation file: the reporter uses that one
# header of the library without linking it, so it only needs the search path.
target_include_directories(squey-crashreport PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
