#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${OPENGL_INCLUDE_DIR})

if (CUDA_FOUND)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)


set(LINK_LIBRARIES pvdisplays picviz pvhive pvkernel ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBRARIES})

# CUDA
set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
set(CUDA_NVCC_FLAGS "-Xcompiler=-fPIC ${CMAKE_NVCC_GENCODE_FLAGS}" "-Xptxas=-v")
#set(CUDA_NVCC_FLAGS "-Xcompiler=-fPIC -gencode arch=compute_30,code=sm_30" "-Xptxas=-v")
#set(CUDA_NVCC_FLAGS "-g -G -Xcompiler=-fPIC -gencode arch=compute_20,code=sm_20" "-Xptxas=-v")

add_definitions(${PICVIZ_DEFINITIONS})
add_definitions(${OpenMP_CXX_FLAGS})

################################################
# Ressource declaration
################################################

SET(SRC_FILES
PVAbstractAxisSlider.cpp
PVAbstractAxisSliders.cpp
PVAbstractRangeAxisSliders.cpp
PVAxisGraphicsItem.cpp
PVAxisLabel.cpp
PVAxisHeader.cpp
PVBCIBuffers.cpp
PVFullParallelScene.cpp
PVFullParallelView.cpp
PVFullParallelViewSelectionRectangle.cpp
PVHitGraphBlocksManager.cpp
PVHitGraphBuffer.cpp
PVHitGraphDataInterface.cpp
PVHitGraphDataOMP.cpp
PVHitGraphSSEHelpers.cpp
#PVHitGraphDataZTRight.cpp
PVLibView.cpp
PVLinesView.cpp
PVParallelView.cpp
PVRenderingPipeline.cpp
PVSelectionAxisSlider.cpp
PVSelectionAxisSliders.cpp
PVSelectionGenerator.cpp
PVSlidersGroup.cpp
PVSlidersManager.cpp
PVTools.cpp
PVZoneRendering.cpp
PVZoneTree.cpp
PVZoneTreeBase.cpp
PVZoneTreeNoAlloc.cpp
PVZonesManager.cpp
PVZonesProcessor.cpp
PVZoomAxisSlider.cpp
PVZoomAxisSliders.cpp
PVZoomedParallelScene.cpp
PVZoomedParallelView.cpp
PVZoomedParallelViewSelectionLine.cpp
PVZoomedParallelViewParamsWidget.cpp
PVZoomedSelectionAxisSlider.cpp
PVZoomedSelectionAxisSliders.cpp
PVZoomedZoneTree.cpp
simple_lines_float_view.cpp
simple_lines_int_view.cpp

# Display plugins
PVDisplayViewFullParallel.cpp
PVDisplayViewZoomedParallel.cpp
PVDisplayViewHitCount.cpp
PVDisplayViewScatter.cpp

# ZoomableDrawingArea & co
PVZoomableDrawingArea.cpp
PVZoomableDrawingAreaWithAxes.cpp

PVSelectionRectangle.cpp
PVSelectionRectangleItem.cpp
PVSelectionHandleItem.cpp

PVSelectionRectangleInteractor.cpp

PVZoomableDrawingAreaInteractor.cpp
PVZoomableDrawingAreaInteractorMajorY.cpp

PVZoomableDrawingAreaConstraintsHomothetic.cpp
PVZoomableDrawingAreaConstraintsMajorY.cpp
PVZoomableDrawingAreaInteractorHomothetic.cpp

# views
PVHitCountView.cpp
PVHitCountViewInteractor.cpp
PVHitCountViewParamsWidget.cpp
PVHitCountViewSelectionRectangle.cpp

PVScatterView.cpp
PVScatterViewInteractor.cpp
PVScatterViewDataImpl.cpp
PVScatterViewImage.cpp
PVScatterViewImagesManager.cpp
PVScatterViewDataInterface.cpp
PVScatterViewDataImpl.cpp
PVScatterViewSelectionRectangle.cpp
PVScatterViewParamsWidget.cpp
)

set(CUDA_SRC_FILES
	#PVBCIDrawingEngineCuda.cpp
	PVBCIDrawingBackendCUDA.cpp
)

SET(HEADER_FILES
include/pvparallelview/common.h
include/pvparallelview/PVBCIBuffers.h
include/pvparallelview/PVBCICode.h
include/pvparallelview/PVBCode.h
include/pvparallelview/PVContainerZoneTree.h
include/pvparallelview/PVLinesView.h
include/pvparallelview/PVParallelView.h
include/pvparallelview/PVPODBlockContainer.h
include/pvparallelview/PVQuadTree.h
include/pvparallelview/PVTools.h
include/pvparallelview/PVZoneProcessing.h
include/pvparallelview/PVZonesManager.h
include/pvparallelview/PVZoneTreeBase.h
include/pvparallelview/PVZoneTree.h
include/pvparallelview/PVZoneTreeNoAlloc.h
include/pvparallelview/PVZoomedZoneTree.h
include/pvparallelview/PVZoomedParallelScene.h
include/pvparallelview/PVZoomedParallelView.h
include/pvparallelview/PVSlidersManager.h
include/pvparallelview/PVAxisLabel.h
include/pvparallelview/PVFullParallelView.h
include/pvparallelview/PVFullParallelViewSelectionRectangle.h
include/pvparallelview/PVAxisHeader.h
include/pvparallelview/PVFullParallelScene.h
include/pvparallelview/PVSlidersGroup.h
include/pvparallelview/PVAbstractAxisSlider.h
include/pvparallelview/PVAbstractAxisSliders.h
include/pvparallelview/PVSelectionAxisSlider.h
include/pvparallelview/PVSelectionAxisSliders.h
include/pvparallelview/PVZoomAxisSlider.h
include/pvparallelview/PVZoomAxisSliders.h
include/pvparallelview/PVZoomedSelectionAxisSlider.h
include/pvparallelview/PVZoomedSelectionAxisSliders.h

# ZoomableDrawingArea & co
include/pvparallelview/PVZoomableDrawingArea.h
include/pvparallelview/PVZoomableDrawingAreaWithAxes.h

# views
include/pvparallelview/PVHitCountView.h
include/pvparallelview/PVScatterView.h
include/pvparallelview/PVScatterViewDataInterface.h
include/pvparallelview/PVScatterViewData.h
include/pvparallelview/PVScatterViewDataImpl.h
include/pvparallelview/PVScatterViewImage.h
include/pvparallelview/PVScatterViewImagesManager.h
)

SET(MOC_HEADERS
include/pvparallelview/PVRenderingJob.h
include/pvparallelview/PVFullParallelView.h
include/pvparallelview/PVFullParallelScene.h
include/pvparallelview/PVSelectionRectangleItem.h
include/pvparallelview/PVSelectionRectangle.h
include/pvparallelview/PVZoomedParallelViewSelectionLine.h
include/pvparallelview/PVZonesManager.h
include/pvparallelview/PVZoomedParallelScene.h
include/pvparallelview/PVSlidersGroup.h
include/pvparallelview/PVAxisGraphicsItem.h
include/pvparallelview/PVAxisLabel.h
include/pvparallelview/PVAxisHeader.h
include/pvparallelview/PVAbstractAxisSlider.h
include/pvparallelview/PVAbstractAxisSliders.h
include/pvparallelview/PVSelectionAxisSliders.h
include/pvparallelview/PVZoomAxisSliders.h
include/pvparallelview/PVZoomedSelectionAxisSliders.h
include/pvparallelview/PVZoomedParallelViewParamsWidget.h
include/pvparallelview/PVZoomableDrawingArea.h
include/pvparallelview/PVHitCountView.h
include/pvparallelview/PVHitCountViewParamsWidget.h
include/pvparallelview/PVScatterView.h
include/pvparallelview/PVScatterViewParamsWidget.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

QT5_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT5_WRAP_OPTIONS})

################################################
# Declare the new library and linkage options
################################################

if (CUDA_FOUND)
	cuda_compile(CUDA_KERNELS cuda/bci_cuda_z24.cu)
	set(SRC_FILES ${SRC_FILES} ${CUDA_SRC_FILES} ${CUDA_KERNELS})
	set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif (CUDA_FOUND)

add_library(pvparallelview SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC})
set_target_properties(pvparallelview PROPERTIES SOVERSION 1)

target_link_libraries(pvparallelview ${LINK_LIBRARIES})

INSTALL(TARGETS pvparallelview LIBRARY DESTINATION .
                               RUNTIME DESTINATION .)
