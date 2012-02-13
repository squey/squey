# include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

# if(USE_PVGL2)
#   include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpvgl2/src/include)
# else(USE_PVGL2)
#   include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpvgl/src/include)
# endif(USE_PVGL2)

include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/gui-qt/src/include)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/gui-qt/src/widgets/)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/gui-qt/src/widgets/format-builder/include)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/gui-qt/src/widgets/remote-log/include)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/gui-qt/src/widgets/editors/include)
include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/gui-qt/src/widgets/helpers/include)

# include_directories(AFTER ${GLEW_INCLUDE_PATH})
# include_directories(AFTER ${OPENGL_INCLUDE_DIR})
#include_directories(AFTER ${CURL_INCLUDE_DIR})
include_directories(AFTER ${ICU_INCLUDE})

# PICVIZ
add_definitions(${PICVIZ_DEFINES})
# include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})

set(QT_USE_QTNETWORK TRUE)
set(QT_USE_QTOPENGL TRUE)
set(QT_USE_QTWEBKIT TRUE)
set(QT_USE_QTXML TRUE)
include(${QT_USE_FILE})


if(WIN32)
    link_libraries(${QT_QTMAIN_LIBRARY})
	set(WIN32_EXTRA_LIBS "ws2_32.lib")
endif(WIN32)

include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
add_definitions(${PICVIZ_FLAGS})

set(INSPECTOR_ICON "resources////inspector.ico")
include_directories(AFTER ${PVGL_INCLUDE_DIRS})
add_definitions(${PVGL_FLAGS})

#include_directories(AFTER ${GLUT_INCLUDE_DIR})
SET(MOC_HEADERS
include/PVArgumentEditorCreator.h
include/PVArgumentListModel.h
include/PVArgumentListWidget.h
include/PVAxisComputationDlg.h
include/PVAxisPropertiesWidget.h
include/PVAxesCombinationWidget.h
include/PVAxesCombinationDialog.h
include/PVCustomStyle.h
include/PVCustomTitleDialog.h
include/PVExtractorWidget.h
include/PVColorDialog.h
include/PVDualSlider.h
include/PVExpandSelDlg.h
include/PVExportSelectionDialog.h
include/PVFilesTypesSelWidget.h
include/PVFilterWidget.h
include/PVInputTypeMenuEntries.h
include/PVInvestigationWizard.h
include/PVLayerFilterProcessWidget.h
include/PVLayerStackDelegate.h
include/PVLayerStackEventFilter.h
include/PVLayerStackModel.h
include/PVLayerStackView.h
include/PVLayerStackWidget.h
include/PVListColNrawDlg.h
include/PVListingModel.h
include/PVListingView.h
include/PVListingsTabWidget.h
include/PVListingSortFilterProxyModel.h
include/PVMainWindow.h
include/PVMappingPlottingEditDialog.h
include/PVMapWidget.h
include/PVOpenFileDialog.h
include/PVSaveFileDialog.h
include/PVSaveSceneDialog.h
include/PVSerializeOptionsModel.h
include/PVSerializeOptionsWidget.h
include/PVSortFilterProxyModel.h
include/PVStartScreenWidget.h
include/PVStringListChooserWidget.h
include/PVTabSplitter.h
include/PVViewsListingView.h
include/PVViewsListingWidget.h
include/PVViewsModel.h

widgets/PVCheckableComboBox.h

widgets/editors/include/PVAxisIndexEditor.h
widgets/editors/include/PVAxisIndexCheckBoxEditor.h
widgets/editors/include/PVAxesIndexEditor.h
widgets/editors/include/PVCheckBoxEditor.h
widgets/editors/include/PVColorGradientDualSliderEditor.h
widgets/editors/include/PVEnumEditor.h
widgets/editors/include/PVLayerEnumEditor.h
widgets/editors/include/PVPlainTextEditor.h
widgets/editors/include/PVRegexpEditor.h
widgets/editors/include/PVSpinBoxEditor.h
#widgets/editors/include/PVTextEditEditor.h
widgets/editors/include/PVTimeFormatEditor.h

widgets/helpers/include/PVAxisTypeWidget.h
widgets/helpers/include/PVComboBox.h
widgets/helpers/include/PVMappingModeWidget.h
widgets/helpers/include/PVPlottingModeWidget.h

widgets/format-builder/include/PVAxisTagHelp.h
widgets/format-builder/include/PVFormatBuilderWidget.h
widgets/format-builder/include/PVNrawListingModel.h
widgets/format-builder/include/PVNrawListingWidget.h
widgets/format-builder/include/PVXmlTreeView.h 
widgets/format-builder/include/PVXmlDomModel.h
widgets/format-builder/include/PVXmlParamColorDialog.h
widgets/format-builder/include/PVXmlParamComboBox.h
widgets/format-builder/include/PVXmlParamList.h
widgets/format-builder/include/PVXmlParamTextEdit.h
widgets/format-builder/include/PVXmlParamWidget.h
widgets/format-builder/include/PVXmlParamWidgetBoardAxis.h
widgets/format-builder/include/PVXmlParamWidgetBoardFilter.h
widgets/format-builder/include/PVXmlParamWidgetBoardSplitterRegEx.h
widgets/format-builder/include/PVXmlParamWidgetEditorBox.h
widgets/format-builder/include/PVXmlRegValidatorHighLight.h
widgets/format-builder/include/PVXmlTreeItemDelegate.h
widgets/format-builder/include/PVXmlTimeValidatorHighLight.h

# include/logviewer/addmachinedialog.h
# include/logviewer/fileconnectiondialog.h
# include/logviewer/filedownloader.h
# include/logviewer/logviewerwidget.h

#include/geo/GKLocation.h
#include/geo/GKMapView.h
#include/geo/GKMapView_p.h
)

SET(SRC_FILES
#GKMapView.cpp
main.cpp
PVArgumentListModel.cpp
PVArgumentListWidget.cpp
PVAxisComputationDlg.cpp
PVAxisPropertiesWidget.cpp
PVAxesCombinationWidget.cpp
PVAxesCombinationDialog.cpp
PVCustomStyle.cpp
PVCustomTitleDialog.cpp
PVFilesTypesSelWidget.cpp
PVExtractorWidget.cpp
PVExpandSelDlg.cpp
PVColorDialog.cpp
PVDualSlider.cpp
PVExportSelectionDialog.cpp
PVFilterWidget.cpp
PVInvestigationWizard.cpp
PVLayerFilterProcessWidget.cpp
PVLayerStackDelegate.cpp
PVLayerStackEventFilter.cpp
PVLayerStackModel.cpp
PVLayerStackView.cpp
PVLayerStackWidget.cpp
PVListColNrawDlg.cpp
PVListingModel.cpp
PVListingView.cpp
PVListingsTabWidget.cpp
PVListingSortFilterProxyModel.cpp
PVMainWindow.cpp
PVMainWindowMenusActions.cpp
PVMainWindowSlots.cpp
PVMapWidget.cpp
PVMappingPlottingEditDialog.cpp
PVInputTypeMenuEntries.cpp
PVOpenFileDialog.cpp
PVSaveFileDialog.cpp
PVSaveSceneDialog.cpp
PVSerializeOptionsModel.cpp
PVSerializeOptionsWidget.cpp
PVSortFilterProxyModel.cpp
PVStartScreenWidget.cpp
PVStringListChooserWidget.cpp
PVTabSplitter.cpp
PVViewsListingView.cpp
PVViewsListingWidget.cpp
PVViewsModel.cpp

widgets/PVCheckableComboBox.cpp

widgets/editors/PVAxisIndexEditor.cpp
widgets/editors/PVAxisIndexCheckBoxEditor.cpp
widgets/editors/PVAxesIndexEditor.cpp
widgets/editors/PVCheckBoxEditor.cpp
widgets/editors/PVColorGradientDualSliderEditor.cpp
widgets/editors/PVEnumEditor.cpp
widgets/editors/PVLayerEnumEditor.cpp
widgets/editors/PVPlainTextEditor.cpp
widgets/editors/PVRegexpEditor.cpp
widgets/editors/PVSpinBoxEditor.cpp
widgets/editors/PVTimeFormatEditor.cpp
#widgets/editors/PVTextEditEditor.cpp

widgets/format-builder/PVAxisTagHelp.cpp
widgets/format-builder/PVFormatBuilderWidget.cpp
widgets/format-builder/PVNrawListingModel.cpp
widgets/format-builder/PVNrawListingWidget.cpp
widgets/format-builder/PVXmlDomModel.cpp
widgets/format-builder/PVXmlParamColorDialog.cpp
widgets/format-builder/PVXmlParamComboBox.cpp
widgets/format-builder/PVXmlParamList.cpp
widgets/format-builder/PVXmlParamTextEdit.cpp
widgets/format-builder/PVXmlParamWidget.cpp
widgets/format-builder/PVXmlParamWidgetBoardAxis.cpp
widgets/format-builder/PVXmlParamWidgetBoardFilter.cpp
widgets/format-builder/PVXmlParamWidgetBoardSplitterRegEx.cpp
widgets/format-builder/PVXmlParamWidgetEditorBox.cpp
widgets/format-builder/PVXmlRegValidatorHighLight.cpp
widgets/format-builder/PVXmlTreeItemDelegate.cpp
widgets/format-builder/PVXmlTreeView.cpp
widgets/format-builder/PVXmlTimeValidatorHighLight.cpp

widgets/helpers/PVAxisTypeWidget.cpp
widgets/helpers/PVComboBox.cpp
widgets/helpers/PVMappingModeWidget.cpp
widgets/helpers/PVPlottingModeWidget.cpp

# logviewer/addmachinedialog.cpp
# logviewer/fileconnectiondialog.cpp
# logviewer/filedownloader.cpp
# logviewer/logviewerwidget.cpp
)

SET(QRC_FILES
picviz-inspector_resource.qrc
)

SET(UI_FILES
PVAxisComputationDlg.ui
PVAxesCombinationWidget.ui
PVAxesCombinationWidgetSelRange.ui
widgets/format-builder/PVAxisTagHelp.ui
ui/PVViewCorrelationWidget.ui
PVListColNrawDlg.ui
)

QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
QT4_WRAP_UI(UI_SRC ${UI_FILES})
QT4_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})

add_definitions(-DUSE_NEW_INPUT_SYSTEM)

if(PROTECT_PASS)
	set(SET_PROTECT TRUE)
	set(GALVEZ_CFLAGS $ENV{GALVEZ_READFWD_CFLAGS})
	set(GALVEZ_LIBS $ENV{GALVEZ_READFWD_LIBS})
	set(GALVEZ_LDFLAGS $ENV{GALVEZ_READFWD_LDFLAGS})
	set(GALVEZ_GCC $ENV{GCC_FLAGS_READFWD})
endif()

if(SET_PROTECT)
	list(APPEND SRC_FILES gen_table.c)
endif()

if (WIN32)
	add_executable(picviz-inspector WIN32 ${SRC_FILES} ${MOC_SRC} ${UI_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS})
else (WIN32)
	add_executable(picviz-inspector ${SRC_FILES} ${MOC_SRC} ${UI_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS})
endif(WIN32)

if(SET_PROTECT)
	message(INFO " Protection activated !")
	#set_property(SOURCE PVMainWindow.cpp APPEND PROPERTY COMPILE_FLAGS ${GALVEZ_GCC})
	set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_FLAGS ${GALVEZ_GCC})
	add_definitions(${GALVEZ_CFLAGS})
endif()

set_property(TARGET picviz-inspector APPEND PROPERTY LINK_FLAGS ${GALVEZ_LDFLAGS})
target_link_libraries(picviz-inspector ${QT_LIBRARIES} ${PVSDK_LIBRARY} ${PICVIZ_LIBRARY} ${PVKERNEL_LIBRARY} ${PVGL_LIBRARY} ${WIN32_EXTRA_LIBS} ${TBB_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY} ${ICU_LIBRARY} ${GALVEZ_LIBS})

INSTALL(TARGETS picviz-inspector DESTINATION .)

