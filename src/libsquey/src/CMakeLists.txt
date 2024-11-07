#
# @file
#

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
PVMappingProperties.cpp
PVScaled.cpp
PVScaledNrawCache.cpp
PVScalingProperties.cpp
PVRoot.cpp
PVScene.cpp
PVSource.cpp
PVStateMachine.cpp
PVView.cpp
PVCorrelationEngine.cpp
PVRangeSubSampler.cpp

plugins.cpp
)

if (PYTHON_SUPPORT)
	set(SRC_FILES ${SRC_FILES}
	PVPythonInterpreter.cpp
	PVPythonSource.cpp
	PVPythonSelection.cpp
	PVPythonInputDialog.cpp
	)
	set(MOC_HEADERS
	../include/squey/PVPythonInterpreter.h
	)
endif()

set(WIDGETS_MOC_HEADERS
../include/squey/widgets/PVAxisComboBox.h
../include/squey/widgets/PVMappingScalingEditDialog.h
../include/squey/widgets/PVNewLayerDialog.h

../include/squey/widgets/editors/PVAxisIndexCheckBoxEditor.h
../include/squey/widgets/editors/PVAxisIndexEditor.h
../include/squey/widgets/editors/PVLayerEnumEditor.h
../include/squey/widgets/editors/PVOriginalAxisIndexEditor.h
../include/squey/widgets/editors/PVViewRowsSpinBoxEditor.h
../include/squey/widgets/editors/PVZoneIndexEditor.h
)

set(WIDGETS_SRC_FILES
widgets/PVArgumentListWidgetFactory.cpp
widgets/PVAxisComboBox.cpp
widgets/PVAxisTypeWidget.cpp
widgets/PVMappingModeWidget.cpp
widgets/PVMappingScalingEditDialog.cpp
widgets/PVNewLayerDialog.cpp
widgets/PVScalingModeWidget.cpp

widgets/editors/PVAxisIndexCheckBoxEditor.cpp
widgets/editors/PVAxisIndexEditor.cpp
widgets/editors/PVLayerEnumEditor.cpp
widgets/editors/PVOriginalAxisIndexEditor.cpp
widgets/editors/PVViewRowsSpinBoxEditor.cpp
widgets/editors/PVZoneIndexEditor.cpp
)

###############################################################################
# Generate Qt files
###############################################################################
QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} ${WIDGETS_MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})

###############################################################################
# Define libs
###############################################################################

set(QT6_LIBS Qt6Core Qt6Xml Qt6Gui Qt6Widgets)

add_library(pvsquey SHARED ${SRC_FILES} ${WIDGETS_SRC_FILES} ${MOC_SRC})

set_target_properties(pvsquey PROPERTIES SOVERSION 1)

target_link_libraries(pvsquey ${QT6_LIBS} ${TBB_LIBRARIES}
			     ${Boost_LIBRARIES} ${ICU_LIBRARY}
			     ${OpenMP_LIBRARIES} ${SIGCPP_LIBRARIES} pvkernel curl)

###############################################################################
# Install files
###############################################################################

INSTALL(TARGETS pvsquey LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                       RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
