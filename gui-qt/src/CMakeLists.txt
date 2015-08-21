#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${ICU_INCLUDE})

################################################
# QT5 related include and check
################################################

# For QApplication
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

# For QNetworkAccessManager
find_package(Qt5Network REQUIRED)
include_directories(${Qt5Network_INCLUDE_DIRS})

# For QDomElement
find_package(Qt5Xml REQUIRED)
include_directories(${Qt5Xml_INCLUDE_DIRS})

# For QDesktopWidget
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

################################################
# Extra WIN32 lib
################################################

if(WIN32)
    set(WIN32_EXTRA_LIBS "ws2_32.lib")
endif(WIN32)

################################################
# Ressource declaration
################################################

set(INSPECTOR_ICON "resources////inspector.ico")

SET(MOC_HEADERS
include/PVAxisComputationDlg.h
#include/PVAxesCombinationWidget.h
#include/PVAxesCombinationDialog.h
include/PVExtractorWidget.h
include/PVColorDialog.h
include/PVDualSlider.h
include/PVExpandSelDlg.h
include/PVFilesTypesSelWidget.h
#include/PVListColNrawDlg.h
#include/PVListingModel.h
#include/PVListingSortFilterProxyModel.h
include/PVMainWindow.h
#include/PVMappingPlottingEditDialog.h
include/PVSaveDataTreeDialog.h
include/PVSerializeOptionsWidget.h
include/PVStringListChooserWidget.h
#include/PVViewsiListingWidget.h

widgets/PVCheckableComboBox.h

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
widgets/format-builder/include/PVXmlRegValidatorHighLight.h
widgets/format-builder/include/PVXmlTimeValidatorHighLight.h
widgets/format-builder/include/PVOptionsWidget.h

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
PVAxisComputationDlg.cpp
#PVAxesCombinationWidget.cpp
#PVAxesCombinationDialog.cpp
PVCustomStyle.cpp
PVCustomTitleDialog.cpp
PVFilesTypesSelWidget.cpp
PVExtractorWidget.cpp
PVExpandSelDlg.cpp
PVColorDialog.cpp
PVDualSlider.cpp
#PVListColNrawDlg.cpp
#PVListingModel.cpp
#PVListingSortFilterProxyModel.cpp
PVMainWindow.cpp
PVMainWindowMenusActions.cpp
PVMainWindowSlots.cpp
PVMapWidget.cpp
PVSaveDataTreeDialog.cpp
PVSerializeOptionsModel.cpp
PVSerializeOptionsWidget.cpp
PVStringListChooserWidget.cpp
PVViewsModel.cpp

widgets/PVCheckableComboBox.cpp

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
widgets/format-builder/PVOptionsWidget.cpp

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
widgets/format-builder/PVAxisTagHelp.ui
ui/PVViewCorrelationWidget.ui
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})
QT5_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})
QT5_WRAP_UI(UI_SRC ${UI_FILES})

################################################
# Handle licences for picviz-inspector
################################################

add_definitions(-DUSE_NEW_INPUT_SYSTEM)

if(PROTECT_PASS)
	set(SET_PROTECT TRUE)
	set(GALVEZ_CFLAGS $ENV{GALVEZ_READFWD_CFLAGS})
	set(GALVEZ_LIBS $ENV{GALVEZ_READFWD_LIBS})
	set(GALVEZ_LDFLAGS $ENV{GALVEZ_READFWD_LDFLAGS})
	set(GALVEZ_GCC "$ENV{GCC_FLAGS_READFWD}+fake_table_file[${CMAKE_CURRENT_BINARY_DIR}/gen_table.c]")
endif()

if(SET_PROTECT)
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen_table.c "")
	list(APPEND SRC_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen_table.c)
endif()

if (USE_LTO AND SET_PROTECT)
	# Remove LTO for picviz-inspector as it remove the calls made with MAO
	list(REMOVE_ITEM CMAKE_CXX_FLAGS_RELPROTECT ${LTO_CXX_FLAGS})
	list(REMOVE_ITEM CMAKE_SHARED_LINKER_FLAGS_RELPROTECT ${LTO_LINK_FLAGS})
endif()

if (WIN32)
	add_executable(picviz-inspector WIN32 ${SRC_FILES} ${MOC_SRC} ${UI_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS})
else (WIN32)
	add_executable(picviz-inspector ${SRC_FILES} ${MOC_SRC} ${UI_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS})
endif(WIN32)

if(SET_PROTECT)
	message(STATUS "Protection activated !")
	#set_property(SOURCE PVMainWindow.cpp APPEND PROPERTY COMPILE_FLAGS ${GALVEZ_GCC})
	set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_FLAGS ${GALVEZ_GCC})
	add_definitions(${GALVEZ_CFLAGS})
endif()

set_property(TARGET picviz-inspector APPEND PROPERTY LINK_FLAGS ${GALVEZ_LDFLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

set(PV_QT_LIBS Qt5::Widgets Qt5::Network Qt5::Xml Qt5OpenGL Qt5WebEngineWidgets)

target_link_libraries(picviz-inspector ${QT_LIBRARIES} ${PVPARALLELVIEW_LIBRARIES}
    pvguiqt pvparallelview pvhive picviz pvkernel ${WIN32_EXTRA_LIBS}
    ${TBB_LIBRARIES} ${Boost_LIBRARIES_PYTHON} ${PYTHON_LIBRARY} ${ICU_LIBRARY}
    ${GALVEZ_LIBS} ${OpenMP_LIBRARIES} ${PV_QT_LIBS})

INSTALL(TARGETS picviz-inspector DESTINATION .)

