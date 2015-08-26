#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(LIBGUIQT_SRC_FILES
common.cpp
PVAxesCombinationDialog.cpp
PVAxesCombinationModel.cpp
PVAxesCombinationWidget.cpp
PVAxesListModel.cpp
PVHiveDataTreeModel.cpp
PVLayerFilterProcessWidget.cpp
PVLayerStackDelegate.cpp
PVLayerStackModel.cpp
PVLayerStackView.cpp
PVLayerStackWidget.cpp
PVListDisplayDlg.cpp
PVAbstractListStatsDlg.cpp
PVCountByStringsDlg.cpp
PVListingModel.cpp
PVListingSortFilterProxyModel.cpp
PVListingView.cpp
PVOpenWorkspacesWidget.cpp
PVRootTreeModel.cpp
PVRootTreeView.cpp
PVSortFilterProxyModel.cpp
PVStringSortProxyModel.cpp
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
PVToolTipDelegate.cpp
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
include/pvguiqt/PVAxesListModel.h
include/pvguiqt/PVLayerFilterProcessWidget.h
include/pvguiqt/PVLayerStackDelegate.h
include/pvguiqt/PVLayerStackModel.h
include/pvguiqt/PVLayerStackView.h
include/pvguiqt/PVLayerStackWidget.h
include/pvguiqt/PVListDisplayDlg.h
include/pvguiqt/PVAbstractListStatsDlg.h
include/pvguiqt/PVListingModel.h
include/pvguiqt/PVListingSortFilterProxyModel.h
include/pvguiqt/PVListingView.h
include/pvguiqt/PVOpenWorkspacesWidget.h
include/pvguiqt/PVRootTreeModel.h
include/pvguiqt/PVRootTreeView.h
include/pvguiqt/PVHiveDataTreeModel.h
include/pvguiqt/PVSortFilterProxyModel.h
include/pvguiqt/PVLogoScene.h
include/pvguiqt/PVWorkspacesTabWidget.h
include/pvguiqt/PVWorkspace.h
include/pvguiqt/PVViewDisplay.h
include/pvguiqt/PVAboutBoxDialog.h
include/pvguiqt/PVProjectsTabWidget.h
include/pvguiqt/PVStartScreenWidget.h
include/pvguiqt/PVImportSourceToProjectDlg.h
include/pvguiqt/PVStatsListingWidget.h
include/pvguiqt/PVToolTipDelegate.h
include/pvguiqt/PVExportSelectionDlg.h
)

SET(UI_FILES
PVAxesCombinationWidget.ui
PVAxesCombinationWidgetSelRange.ui
PVListDisplayDlg.ui
)

SET(QRC_FILES
../../gui-qt/src/picviz-inspector_resource.qrc
)

QT4_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})

set(LIBGUIQT_HEADER_FILES
)

SET(QRC_FILES
)

if(ENABLE_CORRELATION)
	list(APPEND LIBGUIQT_SRC_FILES
	PVCorrelationMenu.cpp
	PVAD2GWidget.cpp
	PVAD2GInteractor.cpp
	)

	list(APPEND LIBGUIQT_MOC_HEADERS
	include/pvguiqt/PVCorrelationMenu.h
	include/pvguiqt/PVAD2GWidget.h
	include/pvguiqt/PVAD2GInteractor.h
	)
endif()

set(SRC_FILES ${LIBGUIQT_SRC_FILES})
set(HEADER_FILES ${LIBGUIQT_HEADER_FILES})
set(MOC_HEADERS ${LIBGUIQT_MOC_HEADERS})
QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
QT4_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})
PV_QT4_WRAP_UI(UI_SRC ${UI_FILES})

PV_QT4_UI_INC_DIR(UI_INC_DIR)
include_directories(AFTER ${UI_INC_DIR})

add_library(pvguiqt SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS} ${UI_FILES})

set_target_properties(pvguiqt PROPERTIES SOVERSION 1)

set(PVGUIQT_LINK_LIBRARIES pvdisplays pvhive picviz pvkernel ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES})

target_link_libraries(pvguiqt ${PVGUIQT_LINK_LIBRARIES})

INSTALL(TARGETS pvguiqt LIBRARY DESTINATION .
                        RUNTIME DESTINATION .)
