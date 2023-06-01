#
# @file
#

include_directories(SYSTEM AFTER ${ICU_INCLUDE})

################################################
# Ressource declaration
################################################

set(SQUEY_ICON "resources////squey.ico")

SET(MOC_HEADERS
../include/PVFilesTypesSelWidget.h
../include/PVMainWindow.h
../include/PVStringListChooserWidget.h

../include/widgets/format-builder/PVFormatBuilderWidget.h
../include/widgets/format-builder/PVNrawListingWidget.h
../include/widgets/format-builder/PVXmlTreeView.h 
../include/widgets/format-builder/PVXmlParamColorDialog.h
../include/widgets/format-builder/PVXmlParamComboBox.h
../include/widgets/format-builder/PVXmlParamList.h
../include/widgets/format-builder/PVXmlParamTextEdit.h
../include/widgets/format-builder/PVXmlParamWidget.h
../include/widgets/format-builder/PVXmlParamWidgetBoardAxis.h
../include/widgets/format-builder/PVXmlParamWidgetBoardFilter.h
../include/widgets/format-builder/PVXmlParamWidgetBoardSplitterRegEx.h
../include/widgets/format-builder/PVXmlRegValidatorHighLight.h
../include/widgets/format-builder/PVXmlTimeValidatorHighLight.h
../include/widgets/format-builder/PVOptionsWidget.h
)

SET(SRC_FILES
main.cpp
PVFilesTypesSelWidget.cpp
PVMainWindow.cpp
PVMainWindowMenusActions.cpp
PVMainWindowSlots.cpp
PVSaveDataTreeDialog.cpp
PVStringListChooserWidget.cpp

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
)

SET(QRC_FILES
squey_resource.qrc
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})
QT6_ADD_RESOURCES(QRC_FILES_SRCS ${QRC_FILES})

################################################
# Handle licences for squey
################################################

add_executable(squey ${SRC_FILES} ${MOC_SRC} ${MOC_HEADERS} ${QRC_FILES_SRCS})

set_property(TARGET squey APPEND PROPERTY LINK_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Widgets_EXECUTABLE_COMPILE_FLAGS}")

set(PV_QT_LIBS Qt6Core Qt6Widgets Qt6Network Qt6Xml Qt6Widgets)

target_link_libraries(squey ssl ${QT_LIBRARIES} ${PVPARALLELVIEW_LIBRARIES}
    pvguiqt pvparallelview pvsquey pvkernel ${TBB_LIBRARIES} ${ICU_LIBRARY}
    ${OpenMP_LIBRARIES} ${PV_QT_LIBS} ${PVLICENSE_LIBRARIES})

INSTALL(TARGETS squey DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
