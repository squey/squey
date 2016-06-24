#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

###############################################################################
# Define ressources
###############################################################################

set(SRC_FILES

common.cpp

PVAxesCombination.cpp
PVAxis.cpp
PVLayer.cpp
PVLayerFilter.cpp
PVLayerStack.cpp
PVLinesProperties.cpp
PVMapped.cpp
PVMapping.cpp 
PVMappingFilter.cpp
PVMappingProperties.cpp 
PVPlotted.cpp
PVPlotting.cpp
PVPlottingFilter.cpp
PVPlottingProperties.cpp
PVRoot.cpp
PVScene.cpp
PVSource.cpp
PVStateMachine.cpp
PVView.cpp
PVCorrelationEngine.cpp

plugins.cpp
)

if (MINESET)
set(SRC_FILES ${SRC_FILES} PVMineset.cpp)
endif()

set(WIDGETS_MOC_HEADERS
include/inendi/widgets/PVMappingModeWidget.h
include/inendi/widgets/PVMappingPlottingEditDialog.h
include/inendi/widgets/PVNewLayerDialog.h
include/inendi/widgets/PVPlottingModeWidget.h

include/inendi/widgets/editors/PVAxisIndexCheckBoxEditor.h
include/inendi/widgets/editors/PVAxisIndexEditor.h
include/inendi/widgets/editors/PVLayerEnumEditor.h
include/inendi/widgets/editors/PVOriginalAxisIndexEditor.h
include/inendi/widgets/editors/PVViewRowsSpinBoxEditor.h
include/inendi/widgets/editors/PVZoneIndexEditor.h
)

set(WIDGETS_SRC_FILES
widgets/PVArgumentListWidgetFactory.cpp
widgets/PVAxisTypeWidget.cpp
widgets/PVMappingModeWidget.cpp
widgets/PVMappingPlottingEditDialog.cpp
widgets/PVNewLayerDialog.cpp
widgets/PVPlottingModeWidget.cpp

widgets/editors/PVAxisIndexCheckBoxEditor.cpp
widgets/editors/PVAxisIndexEditor.cpp
widgets/editors/PVLayerEnumEditor.cpp
widgets/editors/PVOriginalAxisIndexEditor.cpp
widgets/editors/PVViewRowsSpinBoxEditor.cpp
widgets/editors/PVZoneIndexEditor.cpp
)

set(MOC_HEADERS
)

###############################################################################
# Generate Qt files
###############################################################################
QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} ${WIDGETS_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

###############################################################################
# Define libs
###############################################################################

set(QT5_LIBS Qt5Core Qt5Xml Qt5Gui Qt5Widgets)

add_library(inendi SHARED ${SRC_FILES} ${WIDGETS_SRC_FILES} ${MOC_SRC})

set_target_properties(inendi PROPERTIES SOVERSION 1)

target_link_libraries(inendi ${QT5_LIBS} ${TBB_LIBRARIES}
			     ${Boost_LIBRARIES} ${ICU_LIBRARY}
			     ${OpenMP_LIBRARIES} ${OPENGL_LIBRARIES} ${SIGCPP_LIBRARIES} pvkernel curl)

###############################################################################
# Install files
###############################################################################

INSTALL(TARGETS inendi LIBRARY DESTINATION lib
                       RUNTIME DESTINATION lib)
