#
# @file
#

include_directories(SYSTEM AFTER ${OPENGL_INCLUDE_DIR})

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
PVBCIBackendImageQPainter.cpp
PVBCIDrawingBackendQPainter.cpp
PVFullParallelScene.cpp
PVFullParallelView.cpp
PVFullParallelViewParamsWidget.cpp
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
PVZoneRendering.cpp
PVZoneTree.cpp
PVZoneTreeBase.cpp
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

# Display plugins
PVDisplayViewFullParallel.cpp
PVDisplayViewZoomedParallel.cpp
PVDisplayViewHitCount.cpp
PVDisplayViewScatter.cpp
PVDisplayViewTimeseries.cpp

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
PVHitCountViewBackend.cpp
PVHitCountViewInteractor.cpp
PVHitCountViewParamsWidget.cpp
PVHitCountViewSelectionRectangle.cpp

PVScatterView.cpp
PVScatterViewBackend.cpp
PVScatterViewInteractor.cpp
PVScatterViewImage.cpp
PVScatterViewImagesManager.cpp
PVScatterViewDataInterface.cpp
PVScatterViewDataImpl.cpp
PVScatterViewSelectionRectangle.cpp
PVScatterViewParamsWidget.cpp

PVSeriesRendererOpenGL.cpp
PVSeriesRendererOffscreen.cpp
PVSeriesView.cpp
PVSeriesViewWidget.cpp
PVSeriesViewParamsWidget.cpp
PVSeriesViewZoomer.cpp
)

set(OPENCL_SRC_FILES
	PVBCIDrawingBackendOpenCL.cpp
	PVBCIBackendImageOpenCL.cpp
)

SET(MOC_HEADERS
../include/pvparallelview/PVRenderingJob.h
../include/pvparallelview/PVFullParallelView.h
../include/pvparallelview/PVFullParallelScene.h
../include/pvparallelview/PVFullParallelViewParamsWidget.h
../include/pvparallelview/PVSelectionRectangleItem.h
../include/pvparallelview/PVSelectionRectangle.h
../include/pvparallelview/PVZoomedParallelViewSelectionLine.h
../include/pvparallelview/PVZoomedParallelScene.h
../include/pvparallelview/PVSlidersGroup.h
../include/pvparallelview/PVAxisGraphicsItem.h
../include/pvparallelview/PVAxisLabel.h
../include/pvparallelview/PVAxisHeader.h
../include/pvparallelview/PVAbstractAxisSlider.h
../include/pvparallelview/PVAbstractAxisSliders.h
../include/pvparallelview/PVSelectionAxisSliders.h
../include/pvparallelview/PVZoomAxisSliders.h
../include/pvparallelview/PVZoomedSelectionAxisSliders.h
../include/pvparallelview/PVZoomedParallelViewParamsWidget.h
../include/pvparallelview/PVZoomableDrawingArea.h
../include/pvparallelview/PVHitCountView.h
../include/pvparallelview/PVHitCountViewParamsWidget.h
../include/pvparallelview/PVScatterView.h
../include/pvparallelview/PVScatterViewParamsWidget.h
../include/pvparallelview/PVSeriesView.h
../include/pvparallelview/PVSeriesViewWidget.h
../include/pvparallelview/PVSeriesViewParamsWidget.h
../include/pvparallelview/PVSeriesViewZoomer.h
../include/pvparallelview/PVSeriesTreeWidget.h
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})

################################################
# Declare the new library and linkage options
################################################

set(QT6_LIBS Qt6Core Qt6Gui Qt6Widgets Qt6Xml Qt6OpenGL Qt6OpenGLWidgets)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/obfuscate-ocl-kernel.sh ${CMAKE_CURRENT_SOURCE_DIR}/opencl/bci_z24.cl ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/opencl/bci_z24.cl
  )

include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND SRC_FILES ${OPENCL_SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/bci_z24.h)

set(LINK_LIBRARIES pvdisplays inendi pvkernel ${QT6_LIBS} ${OPENGL_LIBRARIES}
	${TBB_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBRARIES} ${OPENCL_LIBRARIES} 
	${PVHWLOC_LIBRARIES} /app/lib/libKF5ItemModels.so)

add_library(pvparallelview SHARED ${SRC_FILES} ${MOC_SRC})
set_target_properties(pvparallelview PROPERTIES SOVERSION 1)

target_link_libraries(pvparallelview ${LINK_LIBRARIES})

INSTALL(TARGETS pvparallelview LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
                               RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
