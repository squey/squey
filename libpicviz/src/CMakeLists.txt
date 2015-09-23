#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

###############################################################################
# Define ressources
###############################################################################

set(CUDA_SRC_FILES
cuda/PVPlotted_create_table_cuda.cu
cuda/PVMapped_create_table_cuda.cu
#cuda/selection.cu
)

set(SRC_FILES
#arguments.cpp 
discover.cpp
#eventline.cpp
#index-array.cpp 
#layer.cpp 
#layer-index-array.cpp 
#layer-stack.cpp 
#line-properties.cpp 

common.cpp

PVAxesCombination.cpp
PVAxis.cpp
PVEventline.cpp
PVIndexArray.cpp
PVLayer.cpp
PVLayerFilter.cpp
PVLayerIndexArray.cpp
PVLayerStack.cpp
PVLinesProperties.cpp
PVMapped.cpp
PVMapping.cpp 
PVMappingFilter.cpp
PVMandatoryMappingFilter.cpp
PVMappingProperties.cpp 
PVPlotted.cpp
PVPlotting.cpp
PVPlottingFilter.cpp
PVPlottingProperties.cpp
PVRoot.cpp
PVScene.cpp
PVSelection.cpp
PVSelectionFilter.cpp
PVSparseSelection.cpp
PVSquareArea.cpp
PVDefaultSortingFunc.cpp
PVSortingFunc.cpp
PVSource.cpp
PVStateMachine.cpp
PVView.cpp

plugins.cpp
#square-area.cpp
type-discovery.cpp 
update-protocol.cpp 
#z-level-array.cpp
)

set(WIDGETS_MOC_HEADERS
include/picviz/widgets/PVMappingModeWidget.h
include/picviz/widgets/PVMappingPlottingEditDialog.h
include/picviz/widgets/PVNewLayerDialog.h
include/picviz/widgets/PVPlottingModeWidget.h

include/picviz/widgets/editors/PVAxesIndexEditor.h
include/picviz/widgets/editors/PVAxisIndexCheckBoxEditor.h
include/picviz/widgets/editors/PVAxisIndexEditor.h
include/picviz/widgets/editors/PVLayerEnumEditor.h
include/picviz/widgets/editors/PVOriginalAxisIndexEditor.h
include/picviz/widgets/editors/PVViewRowsSpinBoxEditor.h
include/picviz/widgets/editors/PVZoneIndexEditor.h
)

set(WIDGETS_SRC_FILES
widgets/PVArgumentListWidgetFactory.cpp
widgets/PVAxisTypeWidget.cpp
widgets/PVMappingModeWidget.cpp
widgets/PVMappingPlottingEditDialog.cpp
widgets/PVNewLayerDialog.cpp
widgets/PVPlottingModeWidget.cpp

widgets/editors/PVAxesIndexEditor.cpp
widgets/editors/PVAxisIndexCheckBoxEditor.cpp
widgets/editors/PVAxisIndexEditor.cpp
widgets/editors/PVLayerEnumEditor.cpp
widgets/editors/PVOriginalAxisIndexEditor.cpp
widgets/editors/PVViewRowsSpinBoxEditor.cpp
widgets/editors/PVZoneIndexEditor.cpp
)

set(MOC_HEADERS
)

if(ENABLE_CORRELATION)
	list(APPEND WIDGETS_SRC_FILES
	widgets/PVAD2GEdgeEditor.cpp
	widgets/PVAD2GFunctionPropertiesWidget.cpp
	widgets/PVAD2GListEdgesWidget.cpp
	widgets/PVAD2GRFFListModel.cpp
	widgets/PVCombinOpDelegate.cpp
	)

	list(APPEND SRC_FILES
	PVAD2GView.cpp
	PVCombiningFunctionView.cpp
	PVTFViewRowFiltering.cpp
	PVTFViewRowFilterFastIPv4.cpp
	PVSelRowFilteringFunction.cpp
	)

	list(APPEND MOC_HEADERS
	include/picviz/widgets/PVAD2GEdgeEditor.h
	include/picviz/widgets/PVAD2GFunctionPropertiesWidget.h
	include/picviz/widgets/PVAD2GListEdgesWidget.h
	include/picviz/widgets/PVAD2GRFFListModel.h
	include/picviz/widgets/PVCombinOpDelegate.h
	)
endif()

###############################################################################
# Generate Qt files
###############################################################################
QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} ${WIDGETS_MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

###############################################################################
# Define libs
###############################################################################
# TODO : Should we remove this part? (cpp still exists)
#if(CUDA_FOUND)
#  set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
#
#  set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -arch=compute_20 -code=sm_20 -g)
#  #set(CUDA_NVCC_FLAGS -Xcompiler -fPIC)
#  cuda_compile(CUDA_GENERATED_FILES ${CUDA_SRC_FILES})
#
##  cuda_add_library(picviz SHARED ${CUDA_GENERATED_FILES})
#  cuda_add_library(picviz SHARED ${SRC_FILES} ${MOC_SRC} ${CUDA_GENERATED_FILES})
#else(CUDA_FOUND)
#  add_library(picviz SHARED ${SRC_FILES} ${MOC_SRC})
#endif(CUDA_FOUND)

set(QT5_LIBS Qt5Core Qt5Xml Qt5Gui Qt5Widgets)

add_library(picviz SHARED ${SRC_FILES} ${WIDGETS_SRC_FILES} ${MOC_SRC})

set_target_properties(picviz PROPERTIES SOVERSION 1)

if(ENABLE_CORRELATION)
        set (TULIP_LIBS ${TULIP_LIBRARY} ${TULIP_QT4_LIBRARY} ${TULIP_OGL_LIBRARY})
endif()

if (NOT WIN32)
	set(MATHS_LIBRARY "-lm")
endif(NOT WIN32)

target_link_libraries(picviz ${QT5_LIBS} ${MATHS_LIBRARY} ${TBB_LIBRARIES}
			     ${Boost_LIBRARIES} ${ICU_LIBRARY}
			     ${OpenMP_LIBRARIES} ${TULIP_LIBS}
			     ${OPENGL_LIBRARIES} pvkernel)

###############################################################################
# Install files
###############################################################################

INSTALL(TARGETS picviz LIBRARY DESTINATION .
                       RUNTIME DESTINATION .)
