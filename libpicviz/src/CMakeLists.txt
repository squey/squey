include_directories(AFTER ${picviz-inspector_SOURCE_DIR}/libpicviz/src/include)

add_definitions(${PICVIZ_DEFINITIONS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${TULIP_INCLUDE_DIR})

if (NOT WIN32)
set(MATHS_LIBRARY "-lm")
endif(NOT WIN32)


set(NONCUDA_HEADER_FILES
	#include/picviz/selection.h
)

set(CUDA_HEADER_FILES
include/picviz/cuda/PVPlotted_create_table_cuda.h
include/picviz/cuda/PVMapped_create_table_cuda.h
include/picviz/cuda/PVCudaDefine.h
	#include/picviz/selection.h
)

set(HEADER_FILES
include/bithacks.h
include/picviz

include/picviz/arguments.h
include/picviz/defs.h
include/picviz/discover.h
include/picviz/eventline.h
include/picviz/general.h
include/picviz/index-array.h
include/picviz/init.h
include/picviz/layer.h
include/picviz/layer-index-array.h
include/picviz/layer-stack.h
include/picviz/limits.h
#include/picviz/line-properties.h
include/picviz/memory.h
include/picviz/plugins.h

include/picviz/PVAD2GView.h
include/picviz/PVAxesCombination.h
include/picviz/PVAxis.h
include/picviz/PVBitset.h
include/picviz/PVEventline.h
include/picviz/PVIndexArray.h
include/picviz/PVLayer.h
include/picviz/PVLayerIndexArray.h
include/picviz/PVLayerFilter.h
include/picviz/PVLinesProperties.h
include/picviz/PVMapped.h
include/picviz/PVMapping.h
include/picviz/PVMappingFilter.h
include/picviz/PVMandatoryMappingFilter.h
include/picviz/PVMappingProperties.h
include/picviz/PVPlotted.h
include/picviz/PVPlotting.h
include/picviz/PVPlottingFilter.h
include/picviz/PVPlottingProperties.h
include/picviz/PVRoot.h
include/picviz/PVScene.h
include/picviz/PVSelection.h
include/picviz/PVSelectionFilter.h
include/picviz/PVSelectionFilterScatterPlotSelectionSquare.h
include/picviz/PVSortQVectorQStringList.h
include/picviz/PVSortQVectorQStringListThread.h
include/picviz/PVDefaultSortingFunc.h
include/picviz/PVSortingFunc.h
include/picviz/PVSource.h
include/picviz/PVSquareArea.h
include/picviz/PVStateMachine.h
include/picviz/PVVector.h
include/picviz/PVView.h
include/picviz/PVZLevelArray.h

# Correlation files
include/picviz/PVSelRowFilteringFunction.h
include/picviz/PVSelRowFilteringFunction_types.h
include/picviz/PVTFViewRowFiltering.h
include/picviz/PVTFViewRowFilterFastIPv4.h
include/picviz/PVTransformationFunctionView.h
include/picviz/PVTransformationFunctionView_types.h

include/picviz/square-area.h
include/picviz/tags.h
include/picviz/time.h
include/picviz/type-discovery.h
include/picviz/z-level-array.h
)

set(NONCUDA_SRC_FILES
#selection.cpp
)

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
init.cpp
#layer.cpp 
#layer-index-array.cpp 
#layer-stack.cpp 
#line-properties.cpp 

PVAD2GView.cpp
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
PVSelectionFilterScatterPlotSelectionSquare.cpp
PVSquareArea.cpp
PVSortQVectorQStringList.cpp
PVSortQVectorQStringListThread.cpp
PVDefaultSortingFunc.cpp
PVSortingFunc.cpp
PVSource.cpp
PVStateMachine.cpp
PVVector.cpp
PVView.cpp
PVZLevelArray.cpp

# Correlation files
PVCombiningFunctionView.cpp
PVTFViewRowFiltering.cpp
PVTFViewRowFilterFastIPv4.cpp

memory.cpp
plugins.cpp
#square-area.cpp
type-discovery.cpp 
update-protocol.cpp 
#z-level-array.cpp
)

set(WIDGETS_MOC_HEADERS
include/picviz/widgets/PVAD2GWidget.h
)

set(WIDGETS_SRC_FILES
widgets/PVAD2GWidget.cpp
)

set(MOC_HEADERS
include/picviz/PVSortQVectorQStringList.h
include/picviz/PVSortQVectorQStringListThread.h
)
QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} ${WIDGETS_MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})

if(CUDA_FOUND)
  set(CUDA_PROPAGATE_HOST_FLAGS "OFF")

  remove_definitions(-DPICVIZ_FUNCTIONS_DIR="${PICVIZ_FUNCTIONS_DIR}")
  remove_definitions(-DPICVIZ_FILTERS_DIR="${PICVIZ_FILTERS_DIR}")
  remove_definitions(-DPICVIZ_LAYER_FILTERS_DIR="${PICVIZ_LAYER_FILTERS_DIR}")
  remove_definitions(-DPICVIZ_MAPPING_FILTERS_DIR="${PICVIZ_MAPPING_FILTERS_DIR}")
  remove_definitions(-DPICVIZ_PLOTTING_FILTERS_DIR="${PICVIZ_PLOTTING_FILTERS_DIR}")
  set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -arch=compute_20 -code=sm_20 -g)
  #set(CUDA_NVCC_FLAGS -Xcompiler -fPIC)
  cuda_compile(CUDA_GENERATED_FILES ${CUDA_SRC_FILES})
  add_definitions(-DPICVIZ_FUNCTIONS_DIR="${PICVIZ_FUNCTIONS_DIR}")
  add_definitions(-DPICVIZ_FILTERS_DIR="${PICVIZ_FILTERS_DIR}")
  add_definitions(-DPICVIZ_LAYER_FILTERS_DIR="${PICVIZ_LAYER_FILTERS_DIR}")
  add_definitions(-DPICVIZ_MAPPING_FILTERS_DIR="${PICVIZ_MAPPING_FILTERS_DIR}")
  add_definitions(-DPICVIZ_PLOTTING_FILTERS_DIR="${PICVIZ_PLOTTING_FILTERS_DIR}")

#  cuda_add_library(picviz SHARED ${CUDA_GENERATED_FILES} ${CUDA_HEADER_FILES})
  cuda_add_library(picviz SHARED ${SRC_FILES} ${WIDGETS_SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS} ${CUDA_GENERATED_FILES})
else(CUDA_FOUND)
  add_library(picviz SHARED ${SRC_FILES} ${WIDGETS_SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS} ${NONCUDA_SRC_FILES} ${NONCUDA_HEADER_FILES})
endif(CUDA_FOUND)
#add_library(picviz STATIC ${SRC_FILES} ${HEADER_FILES})

add_definitions(-DUSE_PARALLEL_MAPPING_FILTERS)
add_definitions(-DUSE_PARALLEL_PLOTTING_FILTERS)
set_target_properties(picviz PROPERTIES SOVERSION 1)

target_link_libraries(picviz ${MATHS_LIBRARY} ${PICVIZ_LIBRARIES_WITHOUT_PICVIZ} ${QT_LIBRARIES} ${PVKERNEL_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${ICU_LIBRARY} ${OpenMP_LIBRARIES} ${TULIP_LIBRARY} ${TULIP_QT4_LIBRARY} ${TULIP_OGL_LIBRARY})

INSTALL(TARGETS picviz LIBRARY DESTINATION .
                       RUNTIME DESTINATION .)
if(WIN32)
	#  INSTALL(FILES ${ZLIB_FILE_DLL} DESTINATION .)
  INSTALL(FILES ${PCRE_FILES_DLL} DESTINATION .)
  #  INSTALL(FILES ${ARCHIVE_FILE_DLL} DESTINATION .)
endif(WIN32)
