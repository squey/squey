#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

################################################
# Ressource declaration
################################################

set(LIBGUIQT_SRC_FILES
common.cpp
PVAxesCombinationDialog.cpp
PVAxesCombinationModel.cpp
PVAxesCombinationWidget.cpp
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
PVLogoModel.cpp
PVLogoScene.cpp
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

# Displays that will be registered to libpvdisplays
PVDisplaySourceDataTree.cpp
PVDisplayViewAxesCombination.cpp
PVDisplayViewLayerStack.cpp
PVDisplayViewListing.cpp
)

SET(LIBGUIQT_MOC_HEADERS
include/pvguiqt/PVAxesCombinationDialog.h
include/pvguiqt/PVAxesCombinationModel.h
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
include/pvguiqt/PVLogoScene.h
include/pvguiqt/PVWorkspacesTabWidget.h
include/pvguiqt/PVWorkspace.h
include/pvguiqt/PVViewDisplay.h
include/pvguiqt/PVAboutBoxDialog.h
include/pvguiqt/PVProjectsTabWidget.h
include/pvguiqt/PVStartScreenWidget.h
include/pvguiqt/PVImportSourceToProjectDlg.h
include/pvguiqt/PVStatsListingWidget.h
include/pvguiqt/PVExportSelectionDlg.h
)

SET(UI_FILES
PVAxesCombinationWidgetSelRange.ui
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
QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})
PV_QT5_WRAP_UI(UI_SRC pvguiqt ${UI_FILES})

################################################
# Declare the new library and linkage options
################################################

add_library(pvguiqt SHARED ${SRC_FILES} ${MOC_SRC} ${QRC_FILES_SRCS} ${UI_FILES})

set_target_properties(pvguiqt PROPERTIES SOVERSION 1)

set(QT5_LIBS Qt5Core Qt5Xml Qt5Widgets Qt5Concurrent Qt5OpenGL)

set(PVGUIQT_LINK_LIBRARIES pvdisplays pvhive inendi pvkernel pvhwloc ${OPENGL_LIBRARIES}
	${TBB_LIBRARIES} ${OpenMP_LIBRARIES})

target_link_libraries(pvguiqt ${PVGUIQT_LINK_LIBRARIES})

INSTALL(TARGETS pvguiqt LIBRARY DESTINATION lib
                        RUNTIME DESTINATION lib)
