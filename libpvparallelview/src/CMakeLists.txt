#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

include_directories(AFTER ${OPENGL_INCLUDE_DIR})

if (CUDA_FOUND)
	include_directories(AFTER ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)


# CUDA
set(CUDA_PROPAGATE_HOST_FLAGS "OFF")
set(CUDA_NVCC_FLAGS "-Xcompiler=-fPIC ${CMAKE_NVCC_GENCODE_FLAGS}" "-Xptxas=-v" "--compiler-bindir gcc")
#set(CUDA_NVCC_FLAGS "-g -G -Xcompiler=-fPIC -gencode arch=compute_20,code=sm_20" "-Xptxas=-v")

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
	PVBCIDrawingBackendCUDA.cpp
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

set(QT5_LIBS Qt5Core Qt5Gui Qt5Widgets Qt5Xml Qt5Concurrent Qt5OpenGL)

set(LINK_LIBRARIES pvdisplays inendi pvhive pvkernel ${QT5_LIBS} ${OPENGL_LIBRARIES}
	${TBB_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBRARIES})

if (CUDA_FOUND)
	cuda_compile(CUDA_KERNELS cuda/bci_cuda_z24.cu)
	set(SRC_FILES ${SRC_FILES} ${CUDA_SRC_FILES} ${CUDA_KERNELS})
	set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif (CUDA_FOUND)

add_library(pvparallelview SHARED ${SRC_FILES} ${MOC_SRC})
set_target_properties(pvparallelview PROPERTIES SOVERSION 1)

target_link_libraries(pvparallelview ${LINK_LIBRARIES})

INSTALL(TARGETS pvparallelview LIBRARY DESTINATION lib
                               RUNTIME DESTINATION lib)
