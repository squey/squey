#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(${QT_QTOPENGL_INCLUDE_DIR})

set(LIBGUIQT_SRC_FILES
PVAxesCombinationDialog.cpp
PVAxesCombinationModel.cpp
PVAxesCombinationWidget.cpp
PVAxesListModel.cpp
PVHiveDataTreeModel.cpp
PVLayerStackDelegate.cpp
PVLayerStackModel.cpp
PVLayerStackView.cpp
PVLayerStackWidget.cpp
PVListingModel.cpp
PVListingSortFilterProxyModel.cpp
PVListingView.cpp
PVRootTreeModel.cpp
PVRootTreeView.cpp
PVSortFilterProxyModel.cpp
PVLogoModel.cpp
PVLogoScene.cpp
)

SET(LIBGUIQT_MOC_HEADERS
include/pvguiqt/PVAxesCombinationDialog.h
include/pvguiqt/PVAxesCombinationModel.h
include/pvguiqt/PVAxesCombinationWidget.h
include/pvguiqt/PVAxesListModel.h
include/pvguiqt/PVLayerStackDelegate.h
include/pvguiqt/PVLayerStackModel.h
include/pvguiqt/PVLayerStackView.h
include/pvguiqt/PVLayerStackWidget.h
include/pvguiqt/PVListingModel.h
include/pvguiqt/PVListingSortFilterProxyModel.h
include/pvguiqt/PVListingView.h
include/pvguiqt/PVHiveDataTreeModel.h
include/pvguiqt/PVSortFilterProxyModel.h
include/pvguiqt/PVLogoScene.h
)

SET(UI_FILES
PVAxesCombinationWidget.ui
PVAxesCombinationWidgetSelRange.ui
)

set(LIBGUIQT_HEADER_FILES
)

SET(QRC_FILES
)

set(SRC_FILES ${LIBGUIQT_SRC_FILES})
set(HEADER_FILES ${LIBGUIQT_HEADER_FILES})
set(MOC_HEADERS ${LIBGUIQT_MOC_HEADERS})
QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
QT4_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})
QT4_WRAP_UI(UI_SRC ${UI_FILES})

add_library(pvguiqt SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS} ${UI_FILES})

set_target_properties(pvguiqt PROPERTIES SOVERSION 1)

set(PVGUIQT_LINK_LIBRARIES ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${PVKERNEL_LIBRARIES} ${PICVIZ_LIBRARIES} ${PVHIVE_LIBRARIES} ${OpenMP_LIBRARIES})

target_link_libraries(pvguiqt ${PVHIVE_LINK_LIBRARIES})

INSTALL(TARGETS pvguiqt LIBRARY DESTINATION .
                        RUNTIME DESTINATION .)
